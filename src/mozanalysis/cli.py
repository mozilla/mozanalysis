from nbformat import read, write, NotebookNode, validate, current_nbformat
from nbformat.v4 import new_notebook
from nbconvert.preprocessors import ExecutePreprocessor
import click

from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from mozanalysis.visualization.StatisticsData import StatisticsData, ViewNotFound

# empty import that triggers registry of all plotting functions
import mozanalysis.visualization.statistics
from mozanalysis.visualization.plotters import Dispatch
from mozanalysis.visualization.types import TimeRange
from mozanalysis.visualization.preamble import (
    download_mozanalysis,
    import_plotters,
    colab_check,
    imports,
    make_analysis_basis_header,
    make_period_header,
    make_period_not_found_header,
    make_metric_header,
    make_download_data_headers,
)
from mozanalysis.visualization.experimenter_api import APIData
from typing import List

experiment_slug_option = click.option(
    "--experiment_slug",
    "--experiment-slug",
    help="Experimenter slug of the experiment to visualize analysis for",
    required=True,
)


@click.group()
def cli():
    pass


@cli.command()
@experiment_slug_option
def generate(experiment_slug):
    api_data = APIData(experiment_slug)
    statistics = StatisticsData(experiment_slug)

    notebook = new_notebook()

    cells = []
    # imports
    cells.append(download_mozanalysis)
    cells.append(import_plotters)
    cells.append(imports)
    cells.append(colab_check)

    # downloading data
    download_data_cells = make_download_data_headers(statistics)
    cells.extend(download_data_cells)

    # calling plotters
    for basis in [AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS]:
        cells.append(make_analysis_basis_header(basis))
        for period in [
            AnalysisPeriod.OVERALL,
            AnalysisPeriod.WEEK,
            AnalysisPeriod.DAY,
            AnalysisPeriod.PREENROLLMENT_DAYS_28,
            AnalysisPeriod.WEEK,
        ]:
            cells.extend(handle_period(period, basis, statistics, api_data))

    notebook["cells"] = cells

    validate(notebook)

    write(notebook, f"{experiment_slug}_raw.ipynb", version=current_nbformat)


def handle_period(
    period: AnalysisPeriod,
    basis: AnalysisBasis,
    statistics: StatisticsData,
    api_data: APIData,
) -> List[NotebookNode]:
    cells = []

    try:
        results = statistics.result_types_for_period_and_basis(period, basis)
        cells.append(make_period_header(period))
        for result_type in results:
            metric = result_type.metric
            cells.append(make_metric_header(metric))
            call_plotter_params = [
                period.value,
                metric,
                period,
                basis,
                api_data.reference_branch(),
                api_data.branch_labels(),
            ]
            if period in [AnalysisPeriod.WEEK, AnalysisPeriod.DAY]:
                plot_type = TimeRange.TimeSeries
            else:
                plot_type = TimeRange.OneTime

            for statistic in result_type.statistics:
                new_cells = Dispatch.dispatch(statistic, plot_type, call_plotter_params)
                cells.extend(new_cells)
    except ViewNotFound:
        cells.append(make_period_not_found_header(period))
        pass

    return cells


@cli.command()
@experiment_slug_option
def render(experiment_slug):

    filename = f"{experiment_slug}_raw.ipynb"
    with open(filename) as ff:
        notebook_in = read(ff, current_nbformat)

    ep = ExecutePreprocessor(timeout=600, kernel_name="mozanalysis")

    notebook_out = ep.preprocess(notebook_in)

    write(notebook_out, f"{experiment_slug}.toml")
