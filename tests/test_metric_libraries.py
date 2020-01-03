import mozanalysis.metrics.desktop as mmd


def test_imported_ok():
    assert mmd.active_hours


def sql_lint(sql):
    safewords = [
        # Exceptions to skip linting
    ]
    for w in safewords:
        if w in sql:
            return

    # Check whether a python string template wasn't filled
    assert '{' not in sql
    assert '}' not in sql

    # Check for balanced parentheses
    assert sql.count('(') == sql.count(')')

    # Check for balanced quote marks
    assert sql.count("'") % 2 == 0


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
