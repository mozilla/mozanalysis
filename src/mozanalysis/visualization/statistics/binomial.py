from nbformat import NotebookNode
from textwrap import dedent
from typing import List, Any
from plotters import make_call_plotter, PlotterRegistry, Dispatch
from StatisticsData import StatisticType
from PlotType import PlotType


timeseries_binomial_individual = dedent(
    f"""\
        def timeseries_binomial_individual(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch:str,
                branches:List[str],
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison.isna())
            )].sort_values('window_index')

            observed_windows = set()
            for branch in branches:
                branch_df = individual_df.loc[(individual_df.branch == branch)]
                plt.plot(branch_df.window_index, branch_df.point)
                plt.fill_between(
                    branch_df.window_index, 
                    branch_df.lower,
                    branch_df.upper,
                    alpha = 0.2, 
                    label = branch
                )
                for window in branch_df.window_index:
                    observed_windows.add(window)
            plt.xticks(sorted(list(observed_windows)))
            plt.legend()
            plt.xlabel(analysis_period)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel(metric)
            plt.show()"""
)

timeseries_binomial_difference = dedent(
    f"""\
        def timeseries_binomial_difference(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch: str,
                branches:List[str],
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison == 'difference') & 
                (period_df.comparison_to_branch == reference_branch)
            )].sort_values('window_index')

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
                    alpha = 0.2, 
                    label = branch
                )
                for window in branch_df.window_index:
                    observed_windows.add(window)                    
            
            plt.xticks(sorted(list(observed_windows)))                        
            plt.legend()
            plt.xlabel(analysis_period)
            #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel('Absolute difference in ' + metric + '(treatment - reference)')
            plt.show()
"""
)

timeseries_binomial_relative = dedent(
    f"""\
        def timeseries_binomial_relative(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch: str,
                branches:List[str],                    
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison == 'relative_uplift') & 
                (period_df.comparison_to_branch == reference_branch)
            )].sort_values('window_index')

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
                    alpha = 0.2, 
                    label = branch
                )
                for window in branch_df.window_index:
                    observed_windows.add(window) 
            plt.xticks(sorted(list(observed_windows)))
            plt.legend()
            plt.xlabel(analysis_period)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel('Relative difference in ' + metric + ' (treatment vs reference)')
            plt.show()
"""
)

onetime_binomial_individual = dedent(
    f"""\
        def onetime_binomial_individual(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch:str,
                branches:List[str],
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison.isna())
            )].sort_values('window_index')

            offset = 0
            for branch in branches:
                branch_df = individual_df.loc[(individual_df.branch == branch)]
                plt.plot([offset], branch_df.point)
                plt.fill_between(
                    [offset], 
                    branch_df.lower,
                    branch_df.upper,
                    alpha = 0.2, 
                    label = branch
                )
                offset += 0.1

            plt.xlabel(analysis_period)                
            plt.legend()
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel(metric)
            plt.show()"""
)

onetime_binomial_difference = dedent(
    f"""\
        def onetime_binomial_difference(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch: str,
                branches:List[str],
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison == 'difference') & 
                (period_df.comparison_to_branch == reference_branch)
            )].sort_values('window_index')

            offset = 0
            for branch in branches:
                if branch == reference_branch: 
                    continue
                branch_df = individual_df.loc[(individual_df.branch == branch)]
                plt.plot([offset], branch_df.point)
                plt.fill_between(
                    [offset], 
                    branch_df.lower,
                    branch_df.upper,
                    alpha = 0.2, 
                    label = branch
                )
                offset += 0.1                  
            
            plt.legend()
            plt.xlabel(analysis_period)
            #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel('Absolute difference in ' + metric + '(treatment - reference)')
            plt.show()
"""
)

onetime_binomial_relative = dedent(
    f"""\
        def onetime_binomial_relative(
                period_df: pd.DataFrame,
                metric: str, 
                analysis_period:str,
                analysis_basis:str,
                reference_branch: str,
                branches:List[str],
            ) -> None:

            fig, ax = plt.subplots()
            individual_df = period_df.loc[(
                (period_df.metric == metric) &
                (period_df.statistic == 'binomial') &
                (period_df.analysis_basis == analysis_basis) & 
                (period_df.comparison == 'relative_uplift') & 
                (period_df.comparison_to_branch == reference_branch)
            )].sort_values('window_index')

            offset = 0
            for branch in branches:
                if branch == reference_branch: 
                    continue
                branch_df = individual_df.loc[(individual_df.branch == branch)]
                plt.plot([offset], branch_df.point)
                plt.fill_between(
                    [offset], 
                    branch_df.lower,
                    branch_df.upper,
                    alpha = 0.2, 
                    label = branch
                )
                offset += 0.1                  
            
            plt.legend()
            plt.xlabel(analysis_period)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.ylabel('Absolute difference in ' + metric + '(treatment - reference)')
            plt.show()
"""
)

for plotter in (
    timeseries_binomial_individual,
    timeseries_binomial_difference,
    timeseries_binomial_relative,
    onetime_binomial_individual,
    onetime_binomial_difference,
    onetime_binomial_relative,
):
    PlotterRegistry.register_plotter(plotter)


def dispatch_binomial_timeseries(call_plotter_params: List[Any]) -> List[NotebookNode]:
    cells = []
    for plotter in [
        "timeseries_binomial_individual",
        "timeseries_binomial_difference",
        "timeseries_binomial_relative",
    ]:
        cells.append(make_call_plotter(plotter, *call_plotter_params))
    return cells


Dispatch.register_dispatch(
    StatisticType.binomial, PlotType.TimeSeries, dispatch_binomial_timeseries
)


def dispatch_binomial_onetime(call_plotter_params: List[Any]) -> List[NotebookNode]:
    cells = []
    for plotter in [
        "onetime_binomial_individual",
        "onetime_binomial_difference",
        "onetime_binomial_relative",
    ]:
        cells.append(make_call_plotter(plotter, *call_plotter_params))
    return cells


Dispatch.register_dispatch(
    StatisticType.binomial, PlotType.OneTime, dispatch_binomial_onetime
)
