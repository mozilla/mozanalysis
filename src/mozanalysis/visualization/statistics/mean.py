from typing import List
from mozanalysis.visualization.plotters import Dispatch
from mozanalysis.visualization.StatisticsData import StatisticType
from mozanalysis.visualization.PlotType import PlotType
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def timeseries_mean_individual(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison.isna())
        )
    ].sort_values("window_index")

    observed_windows = set()
    for branch in branches:
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot(branch_df.window_index, branch_df.point)
        plt.fill_between(
            branch_df.window_index,
            branch_df.lower,
            branch_df.upper,
            alpha=0.2,
            label=branch,
        )
        for window in branch_df.window_index:
            observed_windows.add(window)
    plt.xticks(sorted(list(observed_windows)))
    plt.legend()
    plt.xlabel(analysis_period)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel(metric)
    plt.show()


def timeseries_mean_difference(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison == "difference")
            & (period_df.comparison_to_branch == reference_branch)
        )
    ].sort_values("window_index")

    observed_windows = set()
    for branch in branches:
        if branch == reference_branch:
            continue
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot(branch_df.window_index, branch_df.point)
        plt.fill_between(
            branch_df.window_index,
            branch_df.lower,
            branch_df.upper,
            alpha=0.2,
            label=branch,
        )
        for window in branch_df.window_index:
            observed_windows.add(window)

    plt.xticks(sorted(list(observed_windows)))
    plt.legend()
    plt.xlabel(analysis_period)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Absolute difference in " + metric + "(treatment - reference)")
    plt.show()


def timeseries_mean_relative(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison == "relative_uplift")
            & (period_df.comparison_to_branch == reference_branch)
        )
    ].sort_values("window_index")

    observed_windows = set()
    for branch in branches:
        if branch == reference_branch:
            continue
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot(branch_df.window_index, branch_df.point)
        plt.fill_between(
            branch_df.window_index,
            branch_df.lower,
            branch_df.upper,
            alpha=0.2,
            label=branch,
        )
        for window in branch_df.window_index:
            observed_windows.add(window)
    plt.xticks(sorted(list(observed_windows)))
    plt.legend()
    plt.xlabel(analysis_period)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Relative difference in " + metric + " (treatment vs reference)")
    plt.show()


def onetime_mean_individual(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison.isna())
        )
    ].sort_values("window_index")

    offset = 0
    for branch in branches:
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot([offset], branch_df.point)
        plt.fill_between(
            [offset], branch_df.lower, branch_df.upper, alpha=0.2, label=branch
        )
        offset += 0.1

    plt.xlabel(analysis_period)
    plt.legend()
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel(metric)
    plt.show()


def onetime_mean_difference(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison == "difference")
            & (period_df.comparison_to_branch == reference_branch)
        )
    ].sort_values("window_index")

    offset = 0
    for branch in branches:
        if branch == reference_branch:
            continue
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot([offset], branch_df.point)
        plt.fill_between(
            [offset], branch_df.lower, branch_df.upper, alpha=0.2, label=branch
        )
        offset += 0.1

    plt.legend()
    plt.xlabel(analysis_period)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Absolute difference in " + metric + "(treatment - reference)")
    plt.show()


def onetime_mean_relative(
    period_df: pd.DataFrame,
    metric: str,
    analysis_period: str,
    analysis_basis: str,
    reference_branch: str,
    branches: List[str],
) -> None:

    fig, ax = plt.subplots()
    individual_df = period_df.loc[
        (
            (period_df.metric == metric)
            & (period_df.statistic == "mean")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison == "relative_uplift")
            & (period_df.comparison_to_branch == reference_branch)
        )
    ].sort_values("window_index")

    offset = 0
    for branch in branches:
        if branch == reference_branch:
            continue
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot([offset], branch_df.point)
        plt.fill_between(
            [offset], branch_df.lower, branch_df.upper, alpha=0.2, label=branch
        )
        offset += 0.1

    plt.legend()
    plt.xlabel(analysis_period)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Absolute difference in " + metric + "(treatment - reference)")
    plt.show()


Dispatch.register(
    StatisticType.mean,
    PlotType.TimeSeries,
    [
        timeseries_mean_individual,
        timeseries_mean_difference,
        timeseries_mean_relative,
    ],
)

Dispatch.register(
    StatisticType.mean,
    PlotType.OneTime,
    [
        onetime_mean_individual,
        onetime_mean_difference,
        onetime_mean_relative,
    ],
)
