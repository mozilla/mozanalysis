import pandas as pd
import pytest
from helpers.cheap_lint import sql_lint  # local helper file
from helpers.config_loader_lists import desktop_metrics, desktop_segments

from mozanalysis.config import ConfigLoader
from mozanalysis.experiment import TimeLimits
from mozanalysis.metrics import DataSource, Metric
from mozanalysis.segments import Segment, SegmentDataSource
from mozanalysis.sizing import HistoricalTarget


class DumbResponse:
    """mock class that is returned by the DumbBQ run_query"""

    @staticmethod
    def to_dataframe():
        """Returns empty dataframe with columns corresponeding to metrics
        tested in test_mixed_metric, test_target_metric_mismatch
        and test_target_metric_mismatch_with_custom"""
        return pd.DataFrame(columns=["active_hours", "baseline_ping_count", "qcdou"])


class DumbBQ:
    def run_query(self, *args, **kwargs):
        """returns a DumbResponse object"""
        return DumbResponse()

    def fully_qualify_table_name(self, *args, **kwargs):
        """just exists so an error isn't thrown"""
        return


def test_mixed_metric():
    # NOTE: no equivalent of this test for targets because only
    # desktop currently has segements defined in metric-hub
    bq_context = DumbBQ()

    ht = HistoricalTarget(
        experiment_name="my_test_name",
        start_date="2021-01-01",
        num_dates_enrollment=2,
        analysis_length=4,
    )

    active_hours = ConfigLoader.get_metric("active_hours", "firefox_desktop")
    baseline_ping_count = ConfigLoader.get_metric(
        "baseline_ping_count", "focus_android"
    )

    allweek_regular_v1 = ConfigLoader.get_segment(
        "allweek_regular_v1", "firefox_desktop"
    )

    with pytest.warns(match="metric_list contains multiple metric-hub apps"):
        _ = ht.get_single_window_data(
            bq_context,
            metric_list=[active_hours, baseline_ping_count],
            target_list=[allweek_regular_v1],
        )


def test_target_metric_mismatch():
    bq_context = DumbBQ()

    ht = HistoricalTarget(
        experiment_name="my_test_name",
        start_date="2021-01-01",
        num_dates_enrollment=2,
        analysis_length=4,
    )

    baseline_ping_count = ConfigLoader.get_metric(
        "baseline_ping_count", "focus_android"
    )

    allweek_regular_v1 = ConfigLoader.get_segment(
        "allweek_regular_v1", "firefox_desktop"
    )

    with pytest.warns(match="metric_list and target_list metric-hub apps do not match"):
        _ = ht.get_single_window_data(
            bq_context,
            metric_list=[baseline_ping_count],
            target_list=[allweek_regular_v1],
        )


def test_target_metric_mismatch_with_custom():
    """Includes a custom metric, so the metric_list check should pass
    but the target_list and metric_list comparision should still fail"""
    bq_context = DumbBQ()

    ht = HistoricalTarget(
        experiment_name="my_test_name",
        start_date="2021-01-01",
        num_dates_enrollment=2,
        analysis_length=4,
    )

    baseline_ping_count = ConfigLoader.get_metric(
        "baseline_ping_count", "focus_android"
    )
    clients_daily = ConfigLoader.get_data_source(
        "search_clients_daily", "firefox_desktop"
    )
    qcdou = Metric(
        name="qcdou",
        data_source=clients_daily,
        select_expr="""COUNTIF(
    active_hours_sum > 0 AND
    scalar_parent_browser_engagement_total_uri_count_normal_and_private_mode_sum > 0
)""",
    )

    allweek_regular_v1 = ConfigLoader.get_segment(
        "allweek_regular_v1", "firefox_desktop"
    )

    with pytest.warns(match="metric_list and target_list metric-hub apps do not match"):
        _ = ht.get_single_window_data(
            bq_context,
            metric_list=[baseline_ping_count, qcdou],
            target_list=[allweek_regular_v1],
        )


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
        time_limits=tl,
        target_list=[
            ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop"),
            test_seg,
        ],
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
        time_limits=tl,
        target_list=[
            ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop")
        ],
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
        target_list=desktop_segments,
    )

    sql_lint(target_sql)

    desktop_metrics_no_experiment_slug = [
        m for m in desktop_metrics if "experiment_slug" not in m.select_expr
    ]

    metrics_sql = test_target.build_metrics_query(
        time_limits=tl,
        metric_list=desktop_metrics_no_experiment_slug,
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

    assert "custom_query_target_table" in target_sql
    assert "TEST AGG SELECT STATEMENT" not in target_sql


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
