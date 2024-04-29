from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell
from textwrap import dedent
from typing import List
from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisBasis
from mozanalysis.visualization.StatisticsData import StatisticType
from mozanalysis.visualization.PlotType import PlotType
from typing import Dict, Callable, Any, Optional


class _PlotterRegistry:
    def __init__(self):
        self._registry: List[str] = []

    def register_plotter(self, plotter: str):
        self._registry.append(plotter)

    def generate_plotting_cell(self) -> NotebookNode:
        cell_content = ""
        for plotter in self._registry:
            cell_content += plotter
            cell_content += "\n\n"
        cell = new_code_cell(cell_content)
        cell.metadata = {"jupyter": {"source_hidden": True}}
        return cell


PlotterRegistry = _PlotterRegistry()


def make_statistic_not_supported_header(
    statistic: StatisticType, plot_type: Optional[PlotType] = None
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


plotter_params_type = List[Any]
plotter_function_type = Callable[[plotter_params_type], List[NotebookNode]]
dispatch_function_type = Callable[[plotter_function_type], None]


class _Dispatch:
    def __init__(self):
        self._registry: Dict[StatisticType, Dict[PlotType, plotter_function_type]] = (
            dict()
        )

    # def register_dispatch(
    #     self,
    #     statistic: StatisticType,
    #     plot_type: PlotType,
    #     dispatch_function: Callable[[List[Any]], List[NotebookNode]],
    # ) -> None:
    #     if self._registry.get(statistic) is None:
    #         # first registration for statistic
    #         self._registry[statistic] = {plot_type: dispatch_function}
    #     else:
    #         self._registry[statistic][plot_type] = dispatch_function
    def register(
        self, statistic: StatisticType, plot_type: PlotType
    ) -> Callable[[plotter_function_type], None]:
        def wrap(dispatch_function: dispatch_function_type):
            if self._registry.get(statistic) is None:
                # first registration for statistic
                self._registry[statistic] = {plot_type: dispatch_function}
            else:
                self._registry[statistic][plot_type] = dispatch_function
            return dispatch_function

        return wrap

    def dispatch(
        self,
        statistic: StatisticType,
        plot_type: PlotType,
        call_plotter_params: List[Any],
    ) -> List[NotebookNode]:

        stat_registry = self._registry.get(statistic, None)
        if stat_registry is None:
            return make_statistic_not_supported_header(statistic)

        plotter = stat_registry.get(plot_type, None)
        if plotter is None:
            return make_statistic_not_supported_header(statistic, plot_type)

        return plotter(call_plotter_params)


Dispatch = _Dispatch()


def make_call_plotter(
    plotter_name: str,
    df_name: str,
    metric: str,
    period: AnalysisPeriod,
    basis: AnalysisBasis,
    reference_branch: str,
    branches: List[str],
) -> NotebookNode:
    string = dedent(
        f"""\
            {plotter_name}({df_name}, '{metric}', '{period.value}', '{basis.value}', '{reference_branch}', {branches})"""
    )
    cell = new_code_cell(string)
    return cell
