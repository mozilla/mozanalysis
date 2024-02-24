# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Union, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import re

from mozanalysis.bq import BigQueryContext
from mozanalysis.experiment import TimeSeriesResult
from mozanalysis.metrics import Metric
from mozanalysis.segments import Segment
from mozanalysis.sizing import HistoricalTarget
from mozanalysis.utils import get_time_intervals
from mozanalysis.config import ConfigLoader

from scipy.stats import norm
from math import pi, ceil
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt
import requests


def sample_size_curves(
    df: pd.DataFrame,
    metrics_list: list,
    solver,
    effect_size: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.01,
    power: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.80,
    alpha: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.05,
    **solver_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Loop over a list of different parameters to produce sample size estimates given
    those parameters. A single parameter in [effect_size, power, alpha] should
    be passed a list; the sample size curve will be calculated with this as
    the variable.

    Args:
        df: A pandas DataFrame of queried historical data.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each.
        solver (any function that returns sample size as function of
            effect_size, power, alpha): The solver being used to calculate sample
            size.
        effect_size (float or ArrayLike, default .01): For test of differences in
            proportions, the absolute difference; for tests of differences in mean,
            the percent change.
        alpha (float or ArrayLike, default .05): Significance level for the experiment.
        power (float or ArrayLike, default .90): Probability of detecting an effect,
            when a significant effect exists.
        **solver_kwargs (dict): Arguments necessary for the provided solver.

    Returns:
        A dictionary of pd.DataFrame objects. An item in the dictionary is
        created for each metric in metric_list, containing a DataFrame of sample
        size per branch, number of clients that satisfied targeting, and population
        proportion per branch at each value of the iterable parameter.
    """

    params = {"effect_size": effect_size, "power": power, "alpha": alpha}
    sim_var = [k for k, v in params.items() if type(v) in [list, np.ndarray, pd.Series]]

    if len(sim_var) != 1:
        raise ValueError(
            "Exactly one of effect_size, power, and alpha must be ArrayLike"
        )

    sim_var = sim_var[0]
    test_vals = params[sim_var]
    del params[sim_var]

    results = {}
    for metric in metrics_list:
        sample_sizes = []
        for v in test_vals:
            sample_sizes.append(
                {
                    **{sim_var: v},
                    **solver(df, [metric], **{sim_var: v}, **params, **solver_kwargs)[
                        metric.name
                    ],
                }
            )
        results[metric.name] = pd.DataFrame(sample_sizes)

    return results


def difference_of_proportions_sample_size_calc(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> dict:
    """
    Perform sample size calculation for an experiment to test for a
    difference in proportions.

    Args:
        df: A pandas DataFrame of queried historical data.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each
        effect_size (float, default .01): Difference in proportion for the
            minimum detectable effect --
            effect_size = p(event under alt) - p(event under null)
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns:
        A dictionary. Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric.
    """

    def _get_sample_size_col(col):
        p = np.percentile(df[col], q=[outlier_percentile])[0]
        mean = df.loc[df[col] <= p, col].mean()
        p2 = mean + effect_size

        return samplesize_proportions_2indep_onetail(
            diff=effect_size, prop2=p2, power=power, ratio=1, alpha=alpha, value=0
        )

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.0 * (sample_size / len(df))
        results[col] = {
            "sample_size_per_branch": sample_size,
            "population_percent_per_branch": pop_percent,
            "number_of_clients_targeted": len(df),
        }
    return results


def z_or_t_ind_sample_size_calc(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    test: str = "z",
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> dict:
    """
    Perform sample size calculation for an experiment based on independent
    samples t or z tests.

    Args:
        df: A pandas DataFrame of queried historical data.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each
        test (str, default `z`): `z` or `t` to indicate which solver to use
        effect_size (float, default .01): Percent change in metrics
            expected as a result of the experiment treatment
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns:
        A dictionary. Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric.
    """
    tests = {
        "normal": zt_ind_solve_power,
        "z": zt_ind_solve_power,
        "t": tt_ind_solve_power,
    }
    solver = tests[test]

    def _get_sample_size_col(col):
        p = np.percentile(df[col], q=[outlier_percentile])[0]
        sd = df.loc[df[col] <= p, col].std()
        mean = df.loc[df[col] <= p, col].mean()
        es = (effect_size * mean) / sd

        return solver(effect_size=es, alpha=alpha, power=power, nobs1=None)

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.0 * (sample_size / len(df))
        results[col] = {
            "sample_size_per_branch": sample_size,
            "population_percent_per_branch": pop_percent,
            "number_of_clients_targeted": len(df),
        }
    return results


def empirical_effect_size_sample_size_calc(
    res: TimeSeriesResult,
    bq_context: BigQueryContext,
    metric_list: list,
    quantile: float = 0.90,
    power: float = 0.80,
    alpha: float = 0.05,
    parent_distribution: str = "normal",
    plot_effect_sizes: bool = False,
) -> dict:
    """
    Perform sample size calculation with empirical effect size and
    asymptotic approximation of Wilcoxen-Mann-Whitney U Test. Empirical effect size
    is estimated using a quantile of week-to-week changes over
    the course of the study, and the variance in the test statistic is
    estimated as a quantile of weekly variance in metrics. Sample
    size calculation is based on the asymptotic relative efficiency (ARE) of
    the U test to the T test (see Stapleton 2008, pg 266, or
    https://www.psychologie.hhu.de/fileadmin/redaktion/Fakultaeten/
    Mathematisch-Naturwissenschaftliche_Fakultaet/Psychologie/AAP/gpower/GPowerManual.pdf)

    Args:
        res: A TimeSeriesResult, generated by
            mozanalysis.sizing.HistoricalTarget.get_time_series_data.
        bq_context: A mozanalysis.bq.BigQueryContext object that handles downloading
            time series data from BigQuery.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each.
        quantile (float, default .90): Quantile used to calculate the effect size
            as the quantile of week-to-week metric changes and the variance of
            the mean.
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        parent_distribution (str, default "normal"): Distribution of the parent data;
            must be normal, uniform, logistic, or laplace.
        plot_effect_sizes (bool, default False): Whether or not to plot the
            distribution of effect sizes observed in historical data.

    Returns:
        A dictionary. Keys in the dictionary are the metrics column names from
        the DataFrame; values are dictionaries containing the required sample size
        per branch to achieve the desired power for that metric, along with
        additional information.
    """

    def _mann_whitney_solve_sample_size_approximation(
        effect_size, std, alpha=0.05, power=0.8, parent_distribution="normal"
    ):
        """
        Wilcoxen-Mann-Whitney rank sum test sample size calculation,
        based on asymptotic efficiency relative to the t-test.
        """
        rel_effect_size = effect_size / std
        are = {
            "uniform": 1.0,
            "normal": pi / 3.0,
            "logistic": 9.0 / (pi**2),
            "laplace": 2.0 / 3.0,
        }

        if parent_distribution not in are.keys():
            raise ValueError(f"Parent distribution must be in {are.keys()}")

        t_sample_size = tt_ind_solve_power(
            effect_size=rel_effect_size, power=power, alpha=alpha
        )

        return t_sample_size * are[parent_distribution]

    res_mean, pop_size = res.get_aggregated_data(
        bq_context=bq_context, metric_list=metric_list, aggregate_function="AVG"
    )
    res_mean.sort_values(by="analysis_window_start", ascending=True, inplace=True)

    res_std, _ = res.get_aggregated_data(
        bq_context=bq_context, metric_list=metric_list, aggregate_function="STDDEV"
    )

    size_dict = {}

    for m in metric_list:
        res_mean["diff"] = res_mean[m.name].diff().abs()
        if plot_effect_sizes:
            print(f"{m.name}: plotting effect sizes observed in historical data")
            print("Summary statistics")
            print(res_mean["diff"].describe())
            print("Histogram of effect sizes")
            plt.hist(res_mean["diff"], bins=20)
            plt.show()
        m_quantile = res_mean["diff"].quantile(q=quantile, interpolation="nearest")
        m_std = res_std[m.name].quantile(q=quantile, interpolation="nearest")

        effect_size = {
            "value": m_quantile,
            "period_start_day": res_mean.loc[
                res_mean["diff"] == m_quantile, "analysis_window_start"
            ].values[0],
        }
        effect_size_base_period = effect_size["period_start_day"] - 7
        metric_value = {
            "value": res_mean.loc[
                res_mean["analysis_window_start"] == effect_size_base_period, m.name
            ].values[0],
            "period_start_day": effect_size_base_period,
        }
        std = {
            "value": m_std,
            "period_start_day": res_std.loc[
                res_std[m.name] == m_std, "analysis_window_start"
            ].values[0],
        }
        sample_size = _mann_whitney_solve_sample_size_approximation(
            effect_size=effect_size["value"],
            std=std["value"],
            power=power,
            alpha=alpha,
            parent_distribution=parent_distribution,
        )
        size_dict[m.name] = {
            "effect_size": effect_size,
            "mean": metric_value,
            "std_dev": std,
            "relative_effect_size": effect_size["value"] / metric_value["value"],
            "sample_size_per_branch": sample_size,
            "number_of_clients_targeted": pop_size,
            "population_percent_per_branch": 100.0 * (sample_size / pop_size),
        }

    return size_dict  # TODO: add option to return a DataFrame


def poisson_diff_solve_sample_size(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> dict:
    """
    Sample size for test of difference of Poisson rates,
    based on Poisson rate's asymptotic normality.

    Args:
        df: A pandas DataFrame of queried historical data.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each
        test (str, default `z`): `z` or `t` to indicate which solver to use
        effect_size (float, default .01): Percent change in metrics
            expected as a result of the experiment treatment
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns:
        A dictionary. Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric.
    """

    def _get_sample_size_col(col):
        p = np.percentile(df[col], q=[outlier_percentile])[0]
        sd = df.loc[df[col] <= p, col].std()
        mean = df.loc[df[col] <= p, col].mean()
        es = (effect_size * mean) / sd

        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = norm.ppf(power)

        denom = (es / (z_alpha + z_power)) ** 2
        sample_size = (mean + es) / denom
        return sample_size

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.0 * (sample_size / len(df))
        results[col] = {
            "sample_size_per_branch": sample_size,
            "population_percent_per_branch": pop_percent,
            "number_of_clients_targeted": len(df),
        }
    return results


def variable_enrollment_length_sample_size_calc(
    bq_context: BigQueryContext,
    start_date: Union[str, datetime],
    max_enrollment_days: int,
    analysis_length: int,
    metric_list: List[Metric],
    target_list: List[Segment],
    variable_window_length: int = 7,
    experiment_name: Optional[str] = "",
    app_id: Optional[str] = "",
    to_pandas: bool = True,
    **sizing_kwargs,
) -> Dict[str, Union[Dict[str, int], pd.DataFrame]]:
    """
    Sample size calculation over a variable enrollment window. This function
    will fetch a DataFrame with metrics defined in metric_list for a target
    population defined in the target_list over an enrollment window of length
    max_enrollment_days. Sample size calculation is performed
    using clients enrolled in the first variable_window_length dates in
    that max enrollment window; that window is incrementally widened by
    the variable window length and sample size calculation performed again,
    until the last enrollment date is reached.

    Args:
        bq_context: A mozanalysis.bq.BigQueryContext object that handles downloading
            data from BigQuery.
        start_date (str or datetime in %Y-%m-%d format): First date of enrollment for
            sizing job.
        max_enrollment_days (int): Maximum number of dates to consider for the
            enrollment period for the experiment in question.
        analysis_length (int): Number of days to record metrics for each client
            in the experiment in question.
        metric_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size
            calculation for each.
        target_list (list of mozanalysis.segments.Segment): List of segments
            used to identify clients to include in the study.
        variable_window_length (int): Length of the intervals used to extend
            the enrollment period incrementally. Sample sizes are recalculated over
            each variable enrollment period.
        experiment_name (str): Optional name used to name the target and metric
            tables in BigQuery.
        app_id (str): Application that experiment will be run on.
        **sizing_kwargs: Arguments to pass to z_or_t_ind_sample_size_calc

    Returns:
        A dictionary. Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric.
    """

    if variable_window_length > max_enrollment_days:
        raise ValueError(
            "Enrollment window length is larger than the max enrollment length."
        )

    ht = HistoricalTarget(
        start_date=start_date,
        analysis_length=analysis_length,
        num_dates_enrollment=max_enrollment_days,
        experiment_name=experiment_name,
        app_id=app_id,
    )

    df = ht.get_single_window_data(
        bq_context=bq_context, metric_list=metric_list, target_list=target_list
    )

    interval_end_dates = get_time_intervals(
        start_date,
        variable_window_length,
        max_enrollment_days,
    )

    def _for_interval_sample_size_calculation(i):
        df_interval = df.loc[df["enrollment_date"] < interval_end_dates[i]]
        res = z_or_t_ind_sample_size_calc(
            df=df_interval, metrics_list=metric_list, test="t", **sizing_kwargs
        )
        final_res = {}
        for key in res.keys():
            final_res[key] = {
                "enrollment_end_date": interval_end_dates[i],
                **res[key],
            }

        return final_res

    results_dict = {}
    for m in metric_list:
        results_dict[m.name] = []

    for i in range(len(interval_end_dates)):
        res = _for_interval_sample_size_calculation(i)
        for m in metric_list:
            results_dict[m.name].append(res[m.name])

    for m in results_dict.keys():
        results_dict[m] = pd.DataFrame(results_dict[m])

    return results_dict


def get_firefox_release_dates() -> Dict[int, str]:
    """Download historical major release dates for Firefox Desktop.

    Returns a dict of {<version_int>: <iso_date>}
    """
    r = requests.get("https://whattrainisitnow.com/api/firefox/releases/")
    release_dates = r.json()
    major_releases = {}
    for v, d in release_dates.items():
        m = re.fullmatch("(\d+)\.0", v)
        if m:
            major_releases[int(m.group(1))] = d

    return major_releases


### TODO:
### - adapt for Android (cf cross-platform experiment)
### - defaults for display highlighting?
### - tests
### - review docstrings


class SampleSizing:
    """Run sample size calculations for Firefox Desktop over a historical period.

    This helps to simplify the calculation by collecting relevant information,
    passing it through to the sizing tools, and returning pretty-printed output.
    In particular:

    - Given desired experiment length, automatically determines a historical version
      to start from in order to span full experiment period.
    - Allows for running sizing on a sample, handling sampling automatically
    - Computes sizing for multiple relative effect sizes and returns summary stats
    - Runs empirical sizing and returns summary stats

    Target filter updating works in 2 ways. To specify explicitly where filters are added,
    include the clause `AND {historical_targeting}` in the target SQL condition.
    Otherwise, filters will be injected to replace the final occurrence of `AND` in the target expression.

    Currently, only Firefox Desktop is supported.

    experiment_name: name slug for the analysis, used in BQ table names
    bq_context: BigQuery configuration and client
    metrics: list of mozanalysis Metrics to analyze
    targets: list mozanalysis Segments defining user population to include. For sizing,
        only clients in the intersection of all targets are included.
    n_days_observation_max: max number of days the observation period will cover.
        The historical period will be chosen to allow for this observation length,
        in case sizing is computed for different time periods.
    n_days_enrollment: number of days over which clients are enrolled
    n_days_launch_after_version_release: number of days after version release to begin enrollment
    end_date: most recent date the historical period should cover. Set this to push
        the historical period further into the past. If unspecified, will be set to "yesterday".
    alpha: desired significance level of the experiment
    power: desired power of the experiment
    outlier_percentile: metric values higher than this percentile level will be trimmed.
        Specified as a number between 0 and 1. Set to 0 to disable.
    sample_rate: sampling rate to apply to the population, specified as a number between 0 and 1.
        If set, only this proportion of clients will be retained from the targets
        for sizing calculations. Set to 0 to disable.
    """

    def __init__(
        self,
        experiment_name: str,
        bq_context: BigQueryContext,
        metrics: List[Metric],
        targets: List[Segment],
        n_days_observation_max: int,
        n_days_enrollment: int = 7,
        n_days_launch_after_version_release: int = 7,
        end_date: Optional[str] = None,
        alpha: float = 0.05,
        power: float = 0.9,
        outlier_percentile: float = 0.995,
        sample_rate: float = 0.01,
    ):
        self.bq_context = bq_context
        self.experiment_name = experiment_name
        self.metrics = list(metrics)
        self.targets = list(targets)

        self.n_days_observation_max = n_days_observation_max
        self.n_days_enrollment = n_days_enrollment
        self.n_days_launch_after_version_release = n_days_launch_after_version_release
        self.end_date = end_date

        self.alpha = alpha
        self.power = power
        self.outlier_percentile = outlier_percentile
        self.sample_rate = 0 if sample_rate >= 1 else sample_rate

        self.total_historical_period_days = (
            n_days_launch_after_version_release
            + n_days_enrollment
            + n_days_observation_max
        )
        # Will be set after pulling data
        self._population_size = None

        # Sets self.min_major_version, self.min_major_version_date
        self._find_latest_feasible_historical_version()
        self.enrollment_start_date = (
            date.fromisoformat(self.min_major_version_date)
            + timedelta(days=self.n_days_launch_after_version_release)
        ).isoformat()

        print(
            "\nTotal historical time period will include:",
            f"\n- {n_days_launch_after_version_release} days wait after new version release to launch",
            f"\n- {n_days_enrollment} days enrollment",
            f"\n- up to {n_days_observation_max} days observation.",
            f"\nTo accomodate this, the historical period will start from version {self.min_major_version}",
            f"released on {self.min_major_version_date}.",
            f"Enrollment will start on {self.enrollment_start_date}.",
        )
        if self.sample_rate:
            print(
                f"Target population will be sampled at a rate of {self.sample_rate:.0%}."
            )

        # Update segment definitions to use this version and any sampling.
        self._update_targets_for_historical_period()

    def _find_latest_feasible_historical_version(self) -> None:
        """Find latest version matching our timeline."""
        # Allow a gap of 2 days to be on the safe side.
        latest_feasible_date = date.today() - timedelta(days=2)
        if self.end_date:
            end_date = date.fromisoformat(self.end_date)
        else:
            end_date = latest_feasible_date
        if end_date > latest_feasible_date:
            print(
                f"\nRequested end date {end_date.isoformat()} is later than",
                f"the latest feasible date {latest_feasible_date.isoformat()}.",
                f"Using {latest_feasible_date.isoformat()} as the end date.",
            )
            end_date = latest_feasible_date
        start_date_str = (
            end_date - timedelta(days=self.total_historical_period_days)
        ).isoformat()

        # Find the most recent version released prior the desired start date.
        major_releases = get_firefox_release_dates()
        self.min_major_version = max(
            [v for v, d in major_releases.items() if d <= start_date_str]
        )
        self.min_major_version_date = major_releases[self.min_major_version]

    def _update_targets_for_historical_period(self) -> None:
        """Add an additional Segment to the target list to filter on version and sample ID."""
        historical_filter = (
            f"(browser_version_info.major_version >= {self.min_major_version})"
        )
        if self.sample_rate and self.sample_rate < 1:
            max_sample_id = ceil(self.sample_rate * 100)
            historical_filter += f" AND (sample_id < {max_sample_id})"

        historical_target = Segment(
            name="_historical",
            data_source=ConfigLoader.get_segment_data_source(
                "clients_daily", "firefox_desktop"
            ),
            select_expr=f"COALESCE(LOGICAL_AND({historical_filter}), FALSE)",
        )
        self.targets.append(historical_target)
        print("\nTargeting population matching all of:")
        for t in self.targets:
            print(t.select_expr)

    @property
    def population_size(self) -> int:
        """Get the size of the full eligible population defined by the targeting (ignoring sampling).

        This is unset until a sample size calculation is run which results in the data
            getting pulled.
        """
        if not self._population_size:
            print("Population size will be set on running a sample size calculation")
        return self._population_size

    def _print_population_size(self) -> None:
        """Prints a message displaying population size."""
        msg = f"Eligible population size: {self._population_size:,}"
        if self.sample_rate:
            msg += f", sampled at a rate of {self.sample_rate:.0%}"

        print(msg)

    def _set_population_size(self, pop_size: int) -> None:
        """Set the full population size based on the output of a sample size calculation.

        Makes adjustment for sampling.
        """
        if self.sample_rate:
            pop_size = int(pop_size / self.sample_rate)
        self._population_size = pop_size

    def get_historical_single_window_data(
        self, n_days_observation: int
    ) -> pd.DataFrame:
        """Get single window data for the historical period and population represented by
            the date and segment constraints.

        n_days_observation: number of days the analysis period should cover.

        Returns a DataFrame with 1 row per client.
        """
        ht = HistoricalTarget(
            experiment_name=self.experiment_name,
            start_date=self.enrollment_start_date,
            num_dates_enrollment=self.n_days_enrollment,
            analysis_length=n_days_observation,
        )

        return ht.get_single_window_data(
            bq_context=self.bq_context,
            metric_list=self.metrics,
            target_list=self.targets,
        )

    def get_historical_time_series_data(
        self, n_days_observation: int
    ) -> TimeSeriesResult:
        """Get time series data for the historical period and population represented by
            the date and segment constraints.

        n_days_observation: number of days the analysis period should cover.

        Returns a TimeSeriesResult.
        """
        ht = HistoricalTarget(
            experiment_name=self.experiment_name,
            start_date=self.enrollment_start_date,
            num_dates_enrollment=self.n_days_enrollment,
            analysis_length=n_days_observation,
        )

        return ht.get_time_series_data(
            bq_context=self.bq_context,
            metric_list=self.metrics,
            time_series_period="weekly",
            target_list=self.targets,
        )

    def sample_sizes(
        self, rel_effect_sizes: List[float], n_days_observation: List[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sample sizes for multiple effect sizes and observation period lengths.

        rel_effect_sizes: list of relative effect sizes (% change in metrics, float between 0 and 1)
        n_days_observation: list of observation period length in days not exceeding `n_days_observation_max`.
            If unspecified, the single period length `n_days_observation_max` is used.

        Returns 2 DataFrames:
        TODO:
        """
        n_days_observation = n_days_observation or [self.n_days_observation_max]
        for nd in n_days_observation:
            if nd > self.n_days_observation_max:
                raise ValueError(
                    f"Cannot request observation period longer than {self.n_days_observation_max},"
                    "as was originally budgeted for."
                )

        results = []
        stats = []
        for nd in n_days_observation:
            print(
                f"\nSizing for {self.n_days_enrollment} days enrollment and {nd} days observation."
            )
            full_period_data = self.get_historical_single_window_data(nd)
            if not self._population_size:
                self._set_population_size(len(full_period_data))
                self._print_population_size()

            sizing_results = sample_size_curves(
                full_period_data,
                metrics_list=self.metrics,
                solver=z_or_t_ind_sample_size_calc,
                effect_size=rel_effect_sizes,
                power=self.power,
                alpha=self.alpha,
                outlier_percentile=100 * self.outlier_percentile,
            )

            results.append(
                pd.concat(
                    [
                        v.assign(metric=k, n_days_observation=nd)
                        for k, v in sizing_results.items()
                    ]
                )
            )

            stats.append(
                full_period_data[[m.name for m in self.metrics]]
                .agg(
                    [
                        "mean",
                        "std",
                        lambda d: d[d <= d.quantile(self.outlier_percentile)].mean(),
                        lambda d: d[d <= d.quantile(self.outlier_percentile)].std(),
                    ]
                )
                .set_axis(["mean", "std", "mean_trimmed", "std_trimmed"])
                .transpose()
                .assign(n_days_observation=nd)
                .rename_axis(index="metric")
            )

        df_sizing = (
            pd.concat(results)
            .rename(columns={"effect_size": "rel_effect_size"})
            .drop(columns="number_of_clients_targeted")
            .set_index(["metric", "n_days_observation", "rel_effect_size"])
            .sort_index()
        )
        if self.sample_rate:
            df_sizing["population_percent_per_branch"] = (
                df_sizing["population_percent_per_branch"] * self.sample_rate
            )
        df_stats = (
            pd.concat(stats).set_index("n_days_observation", append=True).sort_index()
        )

        return df_sizing, df_stats

    def display_sample_size_results(
        self,
        sizing_result,
        pct=True,
        effect_sizes=None,
        append_stats=False,
        highlight_lessthan=None,
    ):
        """Pretty-print the sample sizing results returned by `sample_sizes_for_duration()`.

        sizing_result: sizing result dict returned by sample_sizes_for_duration()
        pct: should sample sizes be displayed as a percentage of overall population or as an absolute count?
        effect_sizes: effect sizes to display results for. Can be used to subset columns displayed
        append_stats: should summary stats be appended to the table? If so, displays mean & stdev,
            clipped mean & stdev, and relative differences between stat and clipped stat for each.
        highlight_lessthan: list of (<cutoff (float)>, <color (string)>) tuples. Sample size entries less than <cutoff>
            will be highlighted in <color> in the displayed table.
        """
        sampling_str = (
            f"  (sampled at a rate of {sizing_result['sample_rate']:.0%})"
            if sizing_result["sample_rate"]
            else ""
        )
        print(
            f"\nSizing for {sizing_result['n_days_observation']} days observation.",
            f"Eligible population size: {sizing_result['eligible_population_size']:,}{sampling_str}\n",
        )
        subset = "pop_pcts" if pct else "sample_sizes"
        pp = sizing_result[subset]
        if effect_sizes:
            pp = pp[effect_sizes]
        else:
            effect_sizes = pp.columns
        if append_stats:
            stats_df = sizing_result["stats"]
            stats_df["trim_change_mean"] = (
                stats_df["mean_trimmed"] - stats_df["mean"]
            ).abs() / stats_df["mean"]
            stats_df["trim_change_std"] = (
                stats_df["std_trimmed"] - stats_df["std"]
            ).abs() / stats_df["std"]
            pp = pd.concat([pp, stats_df], axis="columns")

        disp = pp.style.format("{:.2f}%" if pct else "{:,.0f}", subset=effect_sizes)
        if append_stats:
            disp = (
                disp.set_properties(
                    subset=stats_df.columns, **{"background-color": "dimgrey"}
                ).format("{:.2%}", subset=["trim_change_mean", "trim_change_std"])
                # highlight large changes in mean because of trimming
                .applymap(
                    lambda x: "color:maroon; font-weight:bold" if x > 0.15 else "",
                    subset=["trim_change_mean"],
                )
            )

        if highlight_lessthan:
            for lim, color in sorted(
                highlight_lessthan, key=lambda x: x[0], reverse=True
            ):
                disp = disp.highlight_between(
                    subset=effect_sizes, color=color, right=lim
                )

        display(disp)

    def empirical_sizing(
        self, quantile: float = 0.9, style: bool = False
    ) -> Union[pd.DataFrame, Styler]:
        """Compute empirical effect sizes based on week-to-week fluctuations over the maximum observation period.

        Currently no trimming is applied.

        quantile: quantile level to use to select empirical effect sizes and standard deviations.

        If `style` is False, returns a DataFrame with 1 row for each metric and columns containing:
          - relative effect size
          - selected values of effect size (difference), baseline mean and standard deviation
          - sample size and eligible population proportion (adjusted for sampling) per branch
          - index of periods for which values were selected by empirical sizing methodology
        If `style` is True, returns a pandas Styler wrapping this DataFrame which will pretty-print
            the results when displayed.
        """
        print(
            f"\nRunning empirical sizing for {self.n_days_enrollment} days enrollment",
            f"and {self.n_days_observation_max} days observation.",
        )

        tsdata = self.get_historical_time_series_data(self.n_days_observation_max)

        sizing_results = empirical_effect_size_sample_size_calc(
            res=tsdata,
            bq_context=self.bq_context,
            metric_list=self.metrics,
            quantile=quantile,
            power=self.power,
            alpha=self.alpha,
        )

        self._set_population_size(
            list(sizing_results.values())[0]["number_of_clients_targeted"]
        )
        if not style:
            # If using styling, message gets printed there instead
            self._print_population_size()

        # Flatten nested dict output and convert to DF
        formatted_results = {}
        for m, r in sizing_results.items():
            metric_result = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk == "period_start_day":
                            vk = "period"
                        metric_result[f"{k}_{vk}"] = vv
                else:
                    metric_result[k] = v
            formatted_results[m] = metric_result

        sizing_df = pd.DataFrame.from_dict(formatted_results, orient="index")
        if self.sample_rate:
            sizing_df["population_percent_per_branch"] = (
                sizing_df["population_percent_per_branch"] * self.sample_rate
            )
        sizing_df = sizing_df[
            [
                "relative_effect_size",
                "effect_size_value",
                "mean_value",
                "std_dev_value",
                "sample_size_per_branch",
                "population_percent_per_branch",
                "effect_size_period",
                "mean_period",
                "std_dev_period",
            ]
        ]

        if style:
            return self.style_empirical_sizing_result(sizing_df)
        return sizing_df

    def style_empirical_sizing_result(
        self, empirical_sizing_df: pd.DataFrame
    ) -> Styler:
        """Pretty-print the DF returned by `empirical_sizing()`.

        Returns a pandas Styler object.
        """
        self._print_population_size()
        print()

        return (
            empirical_sizing_df[
                [
                    "relative_effect_size",
                    "population_percent_per_branch",
                    "sample_size_per_branch",
                    "effect_size_value",
                    "mean_value",
                    "std_dev_value",
                ]
            ]
            .rename(
                columns={
                    "relative_effect_size": "rel_effect_size",
                    "population_percent_per_branch": "branch_pop_pct",
                    "sample_size_per_branch": "branch_sample_size",
                }
            )
            .style.format(thousands=",", precision=4)
            .format(subset="branch_sample_size", thousands=",", precision=0)
            .format("{:.2f}%", subset="branch_pop_pct")
            .format("{:.2%}", subset="rel_effect_size")
        )
