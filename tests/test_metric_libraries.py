from cheap_lint import sql_lint

import mozanalysis.metrics.desktop as mmd


def test_imported_ok():
    assert mmd.active_hours


def test_sql_not_detectably_malformed():
    for m in mmd.__dict__.values():
        if isinstance(m, mmd.Metric):
            sql_lint(m.select_expr.format(experiment_slug='slug'))

    for ds in mmd.__dict__.values():
        if isinstance(ds, mmd.DataSource):
            sql_lint(ds.from_expr)


def test_consistency_of_metric_and_variable_names():
    for name, metric in mmd.__dict__.items():
        if isinstance(metric, mmd.Metric):
            assert name == metric.name, metric
