from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell
from textwrap import dedent
from typing import List
from mozanalysis.visualization.types import Statistic, TimeRange
from typing import Dict, Callable, Any, Optional


def make_statistic_not_supported_header(
    statistic: Statistic, plot_type: Optional[TimeRange] = None
) -> List[NotebookNode]:
    if plot_type is not None:
        string = dedent(
            f"""\
                ### Statistic: {statistic.name} has no plotting function defined for plot_type {plot_type.name}"""
        )
    else:
        string = dedent(
            f"""\
                ### Statistic: {statistic.name} has no plotting functions defined"""
        )
    cell = new_markdown_cell(string)
    return [cell]


PlotterParametersType = List[Any]
CellsType = List[NotebookNode]
PlotterFunctionType = Callable[[PlotterParametersType], List[NotebookNode]]
DispatchFunctionType = Callable[[PlotterFunctionType], None]


class _Dispatch:
    def __init__(self):
        self._registry: Dict[Statistic, Dict[TimeRange, PlotterFunctionType]] = dict()

    def register(
        self, statistic: Statistic, plot_type: TimeRange, plotters
    ) -> Callable[[PlotterFunctionType], None]:
        if self._registry.get(statistic) is None:
            self._registry[statistic] = {plot_type: plotters}
        else:
            self._registry[statistic][plot_type] = plotters

    def dispatch(
        self,
        statistic: Statistic,
        plot_type: TimeRange,
        call_plotter_params: List[Any],
    ) -> List[NotebookNode]:

        stat_registry = self._registry.get(statistic, None)
        if stat_registry is None:
            return make_statistic_not_supported_header(statistic)

        plotters = stat_registry.get(plot_type, None)
        if plotters is None:
            return make_statistic_not_supported_header(statistic, plot_type)

        cells = []
        for plotter in plotters:
            cells.append(make_call_plotter(plotter.__name__, call_plotter_params))
        return cells


Dispatch = _Dispatch()


def make_call_plotter(
    plotter_name: str, plotter_parameters: PlotterParametersType
) -> NotebookNode:
    df_name, metric, period, basis, reference_branch, branches = plotter_parameters
    string = dedent(
        f"""\
            {plotter_name}({df_name}, '{metric}', '{period.value}', '{basis.value}', '{reference_branch}', {branches})"""
    )
    cell = new_code_cell(string)
    return cell
