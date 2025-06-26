from textwrap import dedent

import pytest
from metric_config_parser import AnalysisUnit

from mozanalysis.experiment import TimeLimits
from mozanalysis.metrics import AnalysisBasis, DataSource, Metric


@pytest.mark.parametrize(
    "experiments_column_type", [None, "simple", "native", "glean", "events_stream"]
)
def test_datasource_constructor_succeeds(experiments_column_type):
    DataSource(
        name="foo",
        from_expr="my_table.name",
        experiments_column_type=experiments_column_type,
    )


@pytest.mark.parametrize(
    "analysis_unit", [AnalysisUnit.CLIENT, AnalysisUnit.PROFILE_GROUP]
)
def test_datasource_build_query_analysis_units(analysis_unit):
    ds = DataSource(
        name="foo",
        from_expr="my_table.name",
        experiments_column_type=None,
    )
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=14,
    )
    empty_str = ""
    fddr = tl.first_date_data_required
    lddr = tl.last_date_data_required

    expected_query = f"""SELECT
            e.analysis_id,
            e.branch,
            e.analysis_window_start,
            e.analysis_window_end,
            e.num_exposure_events,
            e.exposure_date,
            {empty_str}
        FROM enrollments e
            LEFT JOIN my_table.name ds
                ON ds.{analysis_unit.value} = e.analysis_id
                AND ds.submission_date BETWEEN '{fddr}' AND '{lddr}'
                AND ds.submission_date BETWEEN
                    DATE_ADD(e.enrollment_date, interval e.analysis_window_start day)
                    AND DATE_ADD(e.enrollment_date, interval e.analysis_window_end day)
                {empty_str}
        GROUP BY
            e.analysis_id,
            e.branch,
            e.num_exposure_events,
            e.exposure_date,
            e.analysis_window_start,
            e.analysis_window_end"""

    query = ds.build_query([], tl, "", None, AnalysisBasis.ENROLLMENTS, analysis_unit)

    assert query == expected_query


@pytest.mark.parametrize(
    ("use_glean_ids", "expected_id"),
    [
        (True, "glean_client_id"),
        (False, "legacy_telemetry_client_id"),
        (None, "client_id"),
    ],
)
def test_datasource_build_query_glean_ids(use_glean_ids, expected_id):
    ds = DataSource(
        name="foo",
        from_expr="my_table.name",
        experiments_column_type=None,
        client_id_column="client_id",
        glean_client_id_column="glean_client_id",
        legacy_client_id_column="legacy_telemetry_client_id",
    )
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=14,
    )
    empty_str = ""
    fddr = tl.first_date_data_required
    lddr = tl.last_date_data_required

    expected_query = f"""SELECT
            e.analysis_id,
            e.branch,
            e.analysis_window_start,
            e.analysis_window_end,
            e.num_exposure_events,
            e.exposure_date,
            {empty_str}
        FROM enrollments e
            LEFT JOIN my_table.name ds
                ON ds.{expected_id} = e.analysis_id
                AND ds.submission_date BETWEEN '{fddr}' AND '{lddr}'
                AND ds.submission_date BETWEEN
                    DATE_ADD(e.enrollment_date, interval e.analysis_window_start day)
                    AND DATE_ADD(e.enrollment_date, interval e.analysis_window_end day)
                {empty_str}
        GROUP BY
            e.analysis_id,
            e.branch,
            e.num_exposure_events,
            e.exposure_date,
            e.analysis_window_start,
            e.analysis_window_end"""

    query = ds.build_query(
        [],
        tl,
        "",
        None,
        AnalysisBasis.ENROLLMENTS,
        AnalysisUnit.CLIENT,
        use_glean_ids=use_glean_ids,
    )

    assert query == expected_query


def test_datasource_build_query_union_metric_rows():
    ds = DataSource(
        name="foo",
        from_expr="my_table.name",
        experiments_column_type=None,
    )

    metrics_list = []
    for i in range(2):
        m = Metric(name=f"test_metric_{i}", data_source=ds, select_expr=f"SELECT {i}")
        metrics_list.append(m)

    expected_query = """
                SELECT
                    * EXCEPT (test_metric_0, test_metric_1),
                    'test_metric_0' AS metric_slug,
                    SAFE_CAST(test_metric_0 AS STRING) AS metric_value
                FROM metrics

UNION ALL

                SELECT
                    * EXCEPT (test_metric_0, test_metric_1),
                    'test_metric_1' AS metric_slug,
                    SAFE_CAST(test_metric_1 AS STRING) AS metric_value
                FROM metrics"""

    query = ds.build_query_union_metric_rows(metrics_list, "test-experiment")

    assert expected_query == dedent(query.rstrip())


@pytest.mark.parametrize(
    ("name", "from_expr", "experiments_column_type", "error"),
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
    with pytest.raises(
        ValueError,
        match="foo: from_expr contains a dataset template but no value was provided.",
    ):
        DataSource(
            name="foo",
            from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        )
    DataSource(
        name="foo",
        from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        default_dataset="dataset",
    )
