from cheap_lint import sql_lint
import pytest

from mozanalysis.metrics import DataSource, Metric
import mozanalysis.metrics.desktop as mmd
import mozanalysis.metrics.fenix as mmf

from . import enumerate_included


def test_imported_ok():
    assert mmd.active_hours
    assert mmf.uri_count


@pytest.fixture()
def included_metrics():
    return enumerate_included((mmd, mmf), Metric)


@pytest.fixture()
def included_datasources():
    return enumerate_included((mmd, mmf), DataSource)


def test_sql_not_detectably_malformed(included_metrics, included_datasources):
    for _, m in included_metrics:
        sql_lint(m.select_expr.format(experiment_slug='slug'))

    for _, ds in included_datasources:
        sql_lint(ds.from_expr)


def test_consistency_of_metric_and_variable_names(included_metrics):
    for name, metric in included_metrics:
        assert name == metric.name, metric


def test_included_metrics_have_docs(included_metrics):
    for _, m in included_metrics:
        assert m.friendly_name, m.name
        assert m.description, m.name
