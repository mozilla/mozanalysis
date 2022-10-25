from cheap_lint import sql_lint

import mozanalysis.metrics.desktop as mad
import mozanalysis.segments.desktop as msd
from mozanalysis.experiment import TimeLimits
from mozanalysis.sizing import HistoricalTarget
from mozanalysis.segments import Segment, SegmentDataSource
from mozanalysis.metrics import Metric, DataSource


def test_multiple_datasource():
    test_target = HistoricalTarget("test_targ", "2022-01-01", 7, 3)

    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2022-01-01",
        last_date_full_data="2022-01-04",
        analysis_start_days=0,
        analysis_length_dates=3,
    )

    test_sds = SegmentDataSource("test_ds", "test_table")
    test_seg = Segment("test_seg", test_sds, "TEST AGG SELECT STATEMENT", "", "")

    target_sql = test_target.build_targets_query(
        time_limits=tl, target_list=[msd.new_unique_profiles, test_seg]
    )

    sql_lint(target_sql)

    assert "ds_1" in target_sql


def test_query_not_detectably_malformed():

    test_target = HistoricalTarget("test_targ", "2022-01-01", 7, 3)

    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2022-01-01",
        last_date_full_data="2022-01-04",
        analysis_start_days=0,
        analysis_length_dates=3,
    )

    target_sql = test_target.build_targets_query(
        time_limits=tl, target_list=[msd.new_unique_profiles]
    )

    sql_lint(target_sql)

    metrics_sql = test_target.build_metrics_query(
        time_limits=tl, metric_list=[], targets_table="targets"
    )

    sql_lint(metrics_sql)


def test_megaquery_not_detectably_malformed():

    test_target = HistoricalTarget("test_targ", "2022-01-01", 7, 3)

    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2022-01-01",
        last_date_full_data="2022-01-04",
        analysis_start_days=0,
        analysis_length_dates=3,
    )

    target_sql = test_target.build_targets_query(
        time_limits=tl,
        target_list=[s for s in msd.__dict__.values() if isinstance(s, Segment)],
    )

    sql_lint(target_sql)

    metrics_sql = test_target.build_metrics_query(
        time_limits=tl,
        metric_list=[
            m
            for m in mad.__dict__.values()
            if isinstance(m, Metric) and "experiment_slug" not in m.select_expr
        ],
        targets_table="targets",
    )

    sql_lint(metrics_sql)


def test_custom_query_override_target():
    test_target = HistoricalTarget("test_targ", "2022-01-01", 7, 3)

    test_sds = SegmentDataSource(
        "test_ds",
        "test_table",
    )

    test_seg = Segment("test_seg", test_sds, "TEST AGG SELECT STATEMENT", "", "")

    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2022-01-01",
        last_date_full_data="2022-01-04",
        analysis_start_days=0,
        analysis_length_dates=3,
    )

    custom_query = """
    SELECT * from custom_query_target_table
    """

    target_sql = test_target.build_targets_query(
        time_limits=tl, target_list=[test_seg], custom_targets_query=custom_query
    )

    assert (
        "custom_query_target_table" in target_sql
        and "TEST AGG SELECT STATEMENT" not in target_sql
    )


def test_resolve_missing_column_names():
    test_ds = DataSource(
        name="test_ds",
        from_expr="SELECT client_id FROM test_set",
        client_id_column=None,
        submission_date_column=None,
    )

    test_metric = Metric(name="test_metric", data_source=test_ds, select_expr="")

    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2022-01-01",
        last_date_full_data="2022-01-13",
        analysis_start_days=0,
        analysis_length_dates=7,
    )

    test_sds = SegmentDataSource(
        name="test_sds",
        from_expr="SELECT * FROM test_sds",
        client_id_column=None,
        submission_date_column=None,
    )

    test_seg = Segment(name="test_seg", data_source=test_sds, select_expr="")

    test_targ = HistoricalTarget(
        experiment_name="test_exp",
        start_date="2022-01-01",
        analysis_length=7,
    )

    targets_sql = test_targ.build_targets_query(time_limits=tl, target_list=[test_seg])

    metrics_sql = test_targ.build_metrics_query(
        time_limits=tl, metric_list=[test_metric], targets_table="test_targ_enrollment"
    )

    assert "None" not in targets_sql
    assert "None" not in metrics_sql
