import pytest

from mozanalysis.metrics import DataSource


@pytest.mark.parametrize("experiments_column_type", [None, "simple", "native"])
def test_datasource_constructor_succeeds(experiments_column_type):
    DataSource(
        name="foo",
        from_expr="my_table.name",
        experiments_column_type=None,
    )


@pytest.mark.parametrize(
    "name,from_expr,experiments_column_type,error",
    [
        (None, "mytable", "simple", TypeError),
        ("name", None, "simple", TypeError),
        ("name", "mytable", "wrong", ValueError),
    ],
)
def test_datasource_constructor_fails(name, from_expr, experiments_column_type, error):
    with pytest.raises(error):
        DataSource(
            name=name,
            from_expr=from_expr,
            experiments_column_type=experiments_column_type,
        )


def test_complains_about_template_without_default():
    with pytest.raises(ValueError):
        DataSource(
            name="foo",
            from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        )
    DataSource(
        name="foo",
        from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        default_dataset="dataset",
    )
