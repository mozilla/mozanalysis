from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell
from textwrap import dedent
from typing import Tuple

from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from mozanalysis.visualization.StatisticsData import StatisticsData, ViewNotFound


def make_download_mozanalysis() -> NotebookNode:
    string = dedent(
        """\
        pip install mozanalysis --quiet"""
    )

    cell = new_code_cell(string)
    return cell


def make_colab_check() -> NotebookNode:
    string = dedent(
        """\
        try:
            import google.colab
            from google.colab import auth
            auth.authenticate_user()
        except:
            pass"""
    )
    cell = new_code_cell(string)
    return cell


def make_imports() -> NotebookNode:
    string = dedent(
        """\
        import attr
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import pandas as pd
        from pandas_gbq import read_gbq
        from typing import List"""
    )

    cell = new_code_cell(string)
    return cell


def make_define_plotters_header() -> NotebookNode:
    string = dedent(
        """\
        # Define plotting functions"""
    )

    cell = new_markdown_cell(string)
    return cell


def make_analysis_basis_header(basis: AnalysisBasis) -> NotebookNode:
    string = dedent(
        f"""\
            # {basis.value} Analysis Basis"""
    )
    cell = new_markdown_cell(string)
    return cell


def make_period_header(period: AnalysisPeriod) -> NotebookNode:
    string = dedent(
        f"""\
            ## Analysis Period: {period.value}"""
    )
    cell = new_markdown_cell(string)
    return cell


def make_period_not_found_header(period: AnalysisPeriod) -> NotebookNode:
    string = dedent(
        f"""\
            ## Analysis Period: {period.value} not found
            Is the analysis ongoing or was there an error? """
    )
    cell = new_markdown_cell(string)
    return cell


def make_metric_header(metric: str) -> NotebookNode:
    string = dedent(
        f"""\
            ### Metric: {metric}"""
    )
    cell = new_markdown_cell(string)
    return cell


def make_download_data_headers(
    statistics: StatisticsData,
) -> Tuple[NotebookNode, NotebookNode]:
    download_data_markdown_str = dedent(
        """\
# Download Data
"""
    )
    download_data_markdown = new_markdown_cell(download_data_markdown_str)

    download_data_header_str = """"""
    for period in AnalysisPeriod:
        try:
            statistics.period_data(period)
        except ViewNotFound:
            continue

        download_data_header_str += dedent(
            f"""\
        {period.value} = read_gbq(
            'SELECT * FROM `{statistics.period_path(period)}`',
            project_id = 'moz-fx-data-bq-data-science'
        )
        """
        )
    download_data_header = new_code_cell(download_data_header_str)

    return download_data_markdown, download_data_header
