from nbformat import NotebookNode
from textwrap import dedent
from typing import List, Any
from mozanalysis.visualization.plotters import (
    make_call_plotter,
    PlotterRegistry,
    Dispatch,
)
from mozanalysis.visualization.StatisticsData import StatisticType
from mozanalysis.visualization.PlotType import PlotType
import matplotlib.pyplot as plt
import pandas as pd


def onetime_empricalcdf_individual(
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
            & (period_df.statistic == "empirical_cdf")
            & (period_df.analysis_basis == analysis_basis)
            & (period_df.comparison.isna())
        )
    ]

    observed_parameters = set()
    for branch in branches:
        branch_df = individual_df.loc[(individual_df.branch == branch)]
        plt.plot(branch_df.parameter, branch_df.point)
        plt.fill_between(
            branch_df.parameter,
            branch_df.lower,
            branch_df.upper,
            alpha=0.2,
            label=branch,
        )
        for parameter in branch_df.parameter:
            observed_parameters.add(parameter)
    plt.xticks(sorted(list(observed_parameters)))
    plt.legend()
    plt.xlabel(analysis_period)

    plt.ylabel(metric)
    plt.show()


def dispatch_empiricalcdf_onetime(call_plotter_params: List[Any]) -> List[NotebookNode]:
    cells = []
    for plotter in ["onetime_empricalcdf_individual"]:
        cells.append(make_call_plotter(plotter, *call_plotter_params))
    return cells


Dispatch.register_dispatch(
    StatisticType.empirical_cdf, PlotType.OneTime, dispatch_empiricalcdf_onetime
)
