# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from scipy.stats import norm
from math import pi
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from mozanalysis.bq import BigQueryContext
from mozanalysis.experiment import TimeSeriesResult
from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail
import numpy as np


def difference_of_proportions_sample_size_calc(
    df,
    metrics_list,
    effect_size=.01,
    alpha=.05,
    power=.90,
    outlier_percentile=99.5
):

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
            minimum detectable effect:
                effect_size = p(event under alt) - p(event under null)
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns a sorted dictionary:
        Keys in the dictionary are the metrics column names from the DataFrame; values
        are the required sample size to achieve the desired power for that metric.
        The dictionary is sorted by required sample size, in descending order.
    """

    def _get_sample_size_col(col):

        p = np.percentile(df[col], q=[outlier_percentile])[0]
        mean = df.loc[df[col] <= p, col].mean()
        p2 = mean + effect_size

        return samplesize_proportions_2indep_onetail(
            diff=effect_size,
            prop2=p2,
            power=power,
            ratio=1,
            alpha=alpha,
            value=0
        )

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.*(sample_size/len(df))
        results[col] = {
            "sample_size": sample_size,
            "population_percent": pop_percent
        }
    return results


def z_or_t_ind_sample_size_calc(
    df,
    metrics_list,
    test="z",
    rel_effect_size=.01,
    alpha=.05,
    power=.90,
    solver=None,
    outlier_percentile=99.5
):

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
        rel_effect_size (float, default .01): Percent change in metrics
            expected as a result of the experiment treatment
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns a sorted dictionary:
        Keys in the dictionary are the metrics column names from the DataFrame; values
        are the required sample size to achieve the desired power for that metric.
        The dictionary is sorted by required sample size, in descending order.
    """
    tests = {
        "normal": zt_ind_solve_power,
        "z": zt_ind_solve_power,
        "t": tt_ind_solve_power
    }
    solver = tests[test]

    def _get_sample_size_col(col):

        p = np.percentile(df[col], q=[outlier_percentile])[0]
        sd = df.loc[df[col] <= p, col].std()
        mean = df.loc[df[col] <= p, col].mean()
        es = (rel_effect_size*mean)/sd

        return solver(
            effect_size=es, alpha=alpha, power=power, nobs1=None
        )

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.*(sample_size/len(df))
        results[col] = {
            "sample_size": sample_size,
            "population_percent": pop_percent
        }
    return results


def empirical_effect_size_sample_size_calc(
    res: TimeSeriesResult,
    bq_context: BigQueryContext,
    metric_list: list,
    quantile: float = .90,
    power: float = .80,
    alpha: float = .05,
    parent_distribution: str = "normal"
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

    Returns a sorted dictionary:
        Keys in the dictionary are the metrics column names from the DataFrame; values
        are the required sample size to achieve the desired power for that metric.
        The dictionary is sorted by required sample size, in descending order.
    """

    def _mann_whitney_solve_sample_size_approximation(
        effect_size,
        std,
        alpha=.05,
        power=.8,
        parent_distribution="normal"
    ):

        """
        Wilcoxen-Mann-Whitney rank sum test sample size calculation,
        based on asymptotic efficiency relative to the t-test.
        """
        rel_effect_size = effect_size/std
        are = {
            "uniform": 1.,
            "normal": pi/3.,
            "logistic": 9./(pi**2),
            "laplace": 2./3.
        }

        if parent_distribution not in are.keys():
            raise ValueError(f"Parent distribution must be in {are.keys()}")

        t_sample_size = tt_ind_solve_power(
            effect_size=rel_effect_size,
            power=power,
            alpha=alpha
        )

        return t_sample_size*are[parent_distribution]

    res_mean, pop_size = res.get_aggregated_data(
        bq_context=bq_context,
        metric_list=metric_list,
        aggregate_function="AVG"
    )

    res_mean.sort_values(by="analysis_window_start", ascending=True, inplace=True)
    effect_size = {
        m.name: res_mean[m.name].diff().abs().quantile(q=quantile) for m in metric_list
        }

    res_std, _ = res.get_aggregated_data(
        bq_context=bq_context,
        metric_list=metric_list,
        aggregate_function="STDDEV"
    )
    std = {m.name: res_std[m.name].quantile(q=quantile) for m in metric_list}

    size_dict = {
        m.name: {
            "effect_size": effect_size[m.name],
            "std_dev": std[m.name],
            "sample_size": _mann_whitney_solve_sample_size_approximation(
                effect_size=effect_size[m.name],
                std=std[m.name],
                power=power,
                alpha=alpha,
                parent_distribution=parent_distribution
            )
        } for m in metric_list
    }

    for k in size_dict.keys():
        size_dict[k]["population_percent"] = 100.*(size_dict[k]["sample_size"]/pop_size)

    return size_dict


def poisson_diff_solve_sample_size(
    df,
    metrics_list,
    rel_effect_size=.01,
    alpha=.05,
    power=.90,
    outlier_percentile=99.5
):
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
        rel_effect_size (float, default .01): Percent change in metrics
            expected as a result of the experiment treatment
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect,
            when a significant effect exists.
        outlier_percentile(float, default .995): Percentile at which to trim
            each columns.

    Returns a dictionary:
        Keys in the dictionary are the metrics column names from the DataFrame; values
        are the required sample size to achieve the desired power for that metric.
    """

    def _get_sample_size_col(col):

        p = np.percentile(df[col], q=[outlier_percentile])[0]
        sd = df.loc[df[col] <= p, col].std()
        mean = df.loc[df[col] <= p, col].mean()
        es = (rel_effect_size*mean)/sd

        z_alpha = norm.ppf(1-alpha/2)
        z_power = norm.ppf(power)

        denom = (es/(z_alpha+z_power))**2
        sample_size = (mean+es)/denom
        return sample_size

    metric_names = [m.name for m in metrics_list]
    results = {}
    for col in metric_names:
        sample_size = _get_sample_size_col(col)
        pop_percent = 100.*(sample_size/len(df))
        results[col] = {
            "sample_size": sample_size,
            "population_percent": pop_percent
        }
    return results