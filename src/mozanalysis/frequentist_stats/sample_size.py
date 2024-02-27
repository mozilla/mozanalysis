# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""module for sample size calculations"""

from typing import List, Union, Dict, Optional
from datetime import datetime
from collections import UserDict
from math import pi

from scipy.stats import norm
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt

from mozanalysis.bq import BigQueryContext
from mozanalysis.experiment import TimeSeriesResult
from mozanalysis.metrics import Metric
from mozanalysis.segments import Segment
from mozanalysis.sizing import HistoricalTarget
from mozanalysis.utils import get_time_intervals


class ResultsHolder(UserDict):
    """
    Object to hold results from different methods.  It extends
    the dictionary objects so that users can interact with it
    with a dictionary with the same keys/values as before, making
    it backward compatible.
    """

    def __init__(self, *args, metrics: dict = None, params: dict = None, **kwargs):
        """
        Args:
            metrics (dict, optional): _description_. Defaults to None.
            params (dict, optional): _description_. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self._metrics = metrics
        self._params = params
        # create this attribute to hold only the results data
        # this dict is what was returned from the sample size
        # methods historically
        self.data = {
            k: v for k, v in self.data.items() if k not in ["metrics", "params"]
        }

    @staticmethod
    def make_friendly_name(ugly_name: str) -> str:
        """Turns a name into a friendly name
        by replacing underscores with spaces and
        capitlizing words other than ones like "per", "of", etc
        Args:
            ugly_name (str): name to make pretty

        Returns:
            pretty_name (str): reformatted name
        """
        keep_all_lowercase = ["per", "of"]
        split_name = ugly_name.split("_")
        split_name = [
            el[0].upper() + el[1:] if el not in keep_all_lowercase else el
            for el in split_name
        ]
        pretty_name = " ".join(split_name)
        return pretty_name


class SampleSizeResultsHolder(ResultsHolder):
    """
    Object to hold results from different methods.  It extends
    the dictionary objects so that users can interact with it
    with a dictionary with the same keys/values as before, making
    it backward compatible.  The dictionary functionality is extended to include
    additional attributes to hold metadata, a method for plotting results and
    a method for returning results as a dataframe
    """

    def get_dataframe(self) -> pd.DataFrame:
        """
        returns data as a dataframe rather than a dict

        Returns:
            (pd.DataFrame): dataframe of results
            The index is the metric and the columns are the outputs from
            the sample size method
        """
        return pd.DataFrame(self.data).transpose()

    def plot_results(self, result_name: str = "sample_size_per_branch"):
        """plots the outputs of the sampling methods

        Args:
            result_name (str): sample size method output to plot.
            Defaults to sample_size_per_branch
        """
        nice_metric_names = [
            (
                el.friendly_name
                if hasattr(el, "friendly_name")
                else self.make_friendly_name(el.name)
            )
            for el in self._metrics
        ]
        nice_metric_map = {
            el.name: friendly_name
            for el, friendly_name in zip(self._metrics, nice_metric_names)
        }
        nice_result = self.make_friendly_name(result_name)
        df = self.get_dataframe().rename(index=nice_metric_map)
        df[result_name].plot(kind="bar")
        plt.ylabel(nice_result)
        plt.xlabel("Metric")
        plt.show(block=False)


class EmpericalEffectSizeResultsHolder(ResultsHolder):
    "ResultsHolder for empirical_effect_size_sample_size_calc"

    def style_empirical_sizing_result(
        self, empirical_sizing_df: pd.DataFrame
    ) -> Styler:
        """Pretty-print the DF returned by `empirical_sizing()`.

        Returns a pandas Styler object.
        """
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

    def get_dataframe(self) -> pd.DataFrame:
        """returns dataframe for results from empirical_effect_size_sample_size_calc

        Arguments:
            style (bool): If true, return a cleaned up and formatted pandas Styler.
            Otherwise return a dataframe
        Returns:
            pd.DataFrame or Styler: dataframe containing results if style is False
            If true it returns a Styler object
        """

        formatted_results = {}
        for m, r in self.data.items():
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

        df = pd.DataFrame.from_dict(formatted_results, orient="index")
        df["effect_size_base_period"] = df["effect_size_period"] - 7
        df["rel_effect_size"] = df["effect_size_value"] / df["mean_value"]

        return df[
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

    def get_styled_dataframe(self) -> Styler:
        """returns styled dataframe for results from
        empirical_effect_size_sample_size_calc

        Arguments:
            style (bool): If true, return a cleaned up and formatted pandas Styler.
            Otherwise return a dataframe
        Returns:
            Styler: styled dataframe for visualization
        """
        df = self.get_dataframe()
        return self.style_empirical_sizing_result(df)


class SampleSizeCurveResultHolder(ResultsHolder):
    def __init__(self, *args, **kwargs):
        """
        Args:
            metrics (dict, optional): _description_. Defaults to None.
            params (dict, optional): _description_. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        # need to modify results so it matches the old format
        self.raw_df = pd.concat(
            [el.set_index("effect_size", append=True) for el in self.data.values()]
        )
        metrics = [el.name for el in self._metrics]
        results_dict = {}
        for el in metrics:
            results_dict[el] = self.raw_df.loc[el, :].reset_index()
        self.data = results_dict

    def set_raw_data_stats(self, input_data: pd.DataFrame) -> pd.DataFrame:
        outlier_percentile = self._params["outlier_percentile"] / 100
        overall_stats = (
            input_data[[m.name for m in self._metrics]]
            .agg(
                [
                    "mean",
                    "std",
                    lambda d: d[d <= d.quantile(outlier_percentile)].mean(),
                    lambda d: d[d <= d.quantile(outlier_percentile)].std(),
                ]
            )
            .transpose()
        )
        overall_stats.columns = ["mean", "std", "mean_trimmed", "std_trimmed"]
        overall_stats["trim_change_mean"] = (
            overall_stats["mean_trimmed"] - overall_stats["mean"]
        ).abs() / overall_stats["mean"]
        overall_stats["trim_change_std"] = (
            overall_stats["std_trimmed"] - overall_stats["std"]
        ).abs() / overall_stats["std"]
        self._raw_data_stats = overall_stats

    def get_dataframe(
        self,
        input_data: pd.DataFrame = None,
        pct: bool = True,
        simulated_values: list = None,
        append_stats: bool = False,
    ) -> pd.DataFrame:
        """
        obtain data for sample size curves

        Args:
            input_data (pd.DataFrame, optional): Input data used to generate results,
                used to generate statistics. Defaults to None.
            pct (bool, optional): If true, percentage of population will be used in
                output rather than raw counts. Defaults to True.
            simulated_values (list, optional): List of simulated values to subset
                output to. If None, all values will be included. Defaults to None.
            append_stats (bool, optional): _description_. If True, summary statistics on
                the raw metrics are provided. Defaults to False.
            highlight_lessthan (list[tuple], optional): List of
                (<cutoff (float)>, <color (string)>) tuples.
                Sample size entries less than <cutoff> will be highlighted in <color>
                in the displayed table. Defaults to None.

        Raises:
            ValueError: Raises error if append_stats is True but no input_data is
                        provided

        Returns:
            pd.DataFrame : dataframe consolidating data
        """
        # choose which column to output
        if pct:
            subset_col = "population_percent_per_branch"
        else:
            subset_col = "sample_size_per_branch"

        # reformate raw_df to make different simulatd values into columns
        if simulated_values is None:
            simulated_values = self._params["simulated_values"]
            pretty_df = self.raw_df[subset_col].unstack()
        else:
            pretty_df = self.raw_df[subset_col].unstack()[simulated_values]

        # get stats if input_data is non-null and append_stats is True
        if append_stats:
            if input_data is None:
                raise ValueError("append_stats is true but no raw data was provided")
            self.set_raw_data_stats(input_data)
            pretty_df = pd.concat([pretty_df, self._raw_data_stats], axis="columns")
        return pretty_df

    def get_styled_dataframe(
        self,
        input_data: pd.DataFrame = None,
        pct: bool = True,
        simulated_values: list = None,
        append_stats: bool = False,
        highlight_lessthan: "list[tuple]" = None,
    ) -> Styler:

        pretty_df = self.get_dataframe(
            input_data,
            pct=pct,
            simulated_values=simulated_values,
            append_stats=append_stats,
        )

        disp = pretty_df.style.format(
            "{:.2f}%" if pct else "{:,.0f}", subset=simulated_values
        )
        if append_stats:
            large_change_format_str = "color:maroon; font-weight:bold"
            disp = (
                disp.set_properties(
                    subset=self._raw_data_stats.columns,
                    **{"background-color": "dimgrey"},
                ).format("{:.2%}", subset=["trim_change_mean", "trim_change_std"])
                # highlight large changes in mean because of trimming
                .applymap(
                    lambda x: large_change_format_str if x > 0.15 else "",
                    subset=["trim_change_mean"],
                )
            )

        if highlight_lessthan:
            for lim, color in sorted(
                highlight_lessthan, key=lambda x: x[0], reverse=True
            ):
                disp = disp.highlight_between(
                    subset=simulated_values, color=color, right=lim
                )

        return disp


def sample_size_curves(
    df: pd.DataFrame,
    metrics_list: list,
    solver,
    effect_size: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.01,
    power: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.80,
    alpha: Union[float, Union[np.ndarray, pd.Series, List[float]]] = 0.05,
    **solver_kwargs,
) -> SampleSizeCurveResultHolder:
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
        SampleSizeCurveResultHolder: The data attribute contains a dictionary
        of pd.DataFrame objects. An item in the dictionary is
        created for each metric in metric_list, containing a DataFrame of sample
        size per branch, number of clients that satisfied targeting, and population
        proportion per branch at each value of the iterable parameter.
        Additional methods for ease of use are documented in the class.
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
    for v in test_vals:
        sample_sizes = solver(
            df, metrics_list, **{sim_var: v}, **params, **solver_kwargs
        ).get_dataframe()
        sample_sizes[sim_var] = v

        results[v] = sample_sizes

    # add sim_var to metadata
    params["sim_var"] = sim_var
    params["simulated_values"] = test_vals
    return SampleSizeCurveResultHolder(
        results, metrics=metrics_list, params={**params, **solver_kwargs}
    )


def difference_of_proportions_sample_size_calc(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> SampleSizeResultsHolder:
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
        SampleSizeResultsHolder: The data attribute contains a dictionary.
        Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric. Additional methods for ease of use
        are documented in the class.
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
    params = {
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
        "outlier_percentile": outlier_percentile,
    }

    return SampleSizeResultsHolder(results, metrics=metrics_list, params=params)


def z_or_t_ind_sample_size_calc(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    test: str = "z",
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> SampleSizeResultsHolder:
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
        SampleSizeResultsHolder: The data attribute contains a dictionary.
        Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric. Additional methods for ease of use
        are documented in the class.
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
    params = {
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
        "outlier_percentile": outlier_percentile,
        "solver": solver,
        "test": test,
    }

    return SampleSizeResultsHolder(results, metrics=metrics_list, params=params)


def empirical_effect_size_sample_size_calc(
    res: TimeSeriesResult,
    bq_context: BigQueryContext,
    metric_list: list,
    quantile: float = 0.90,
    power: float = 0.80,
    alpha: float = 0.05,
    parent_distribution: str = "normal",
    plot_effect_sizes: bool = False,
) -> EmpericalEffectSizeResultsHolder:
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
        EmpericalEffectSizeResultsHolder: The data attribute contains a dictionary.
        Keys in the dictionary are the metrics column names from
        the DataFrame; values are dictionaries containing the required sample size
        per branch to achieve the desired power for that metric, along with
        additional information. Additional methods for ease of use
        are documented in the class.
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

    params = {
        "quantile": quantile,
        "power": power,
        "alpha": alpha,
        "parent_distribution": parent_distribution,
    }

    return EmpericalEffectSizeResultsHolder(
        size_dict, metrics=metric_list, params=params
    )


def poisson_diff_solve_sample_size(
    df: pd.DataFrame,
    metrics_list: List[Metric],
    effect_size: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.90,
    outlier_percentile: float = 99.5,
) -> SampleSizeResultsHolder:
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
        SampleSizeResultsHolder: The data attribute contains a dictionary.
        Keys in the dictionary are the metrics column names from
        the DataFrame; values are the required sample size per branch to achieve
        the desired power for that metric.  Additional methods for ease of use
        are documented in the class
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
    params = {
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
        "outlier_percentile": outlier_percentile,
    }

    return SampleSizeResultsHolder(results, metrics=metrics_list, params=params)


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
