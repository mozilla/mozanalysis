import pytest
from helpers.cheap_lint import sql_lint  # local helper file
from helpers.config_loader_lists import (
    desktop_metrics,
    desktop_segments,
    fenix_metrics,
    firefox_ios_metrics,
    klar_android_metrics,
    klar_ios_metrics,
)
from mozanalysis.config import ApplicationNotFound, ConfigLoader
from mozanalysis.experiment import (
    AnalysisWindow,
    EnrollmentsQueryType,
    Experiment,
    TimeLimits,
)
from mozanalysis.exposure import ExposureSignal
from mozanalysis.metrics import AnalysisBasis, DataSource, Metric
from mozanalysis.segments import Segment, SegmentDataSource


def test_time_limits_validates():
    # Mainly check that the validation is running at all
    # No need to specify the same checks twice(?)
    with pytest.raises(TypeError):
        TimeLimits()

    with pytest.raises(AssertionError):
        TimeLimits(
            first_enrollment_date="2019-01-05",
            last_enrollment_date="2019-01-05",
            analysis_windows=(AnalysisWindow(1, 1),),
            first_date_data_required="2019-01-01",  # Before enrollments
            last_date_data_required="2019-01-01",
        )


def test_time_limits_create1():
    # When we have complete data for 2019-01-14...
    # ...We have 14 dates of data for those who enrolled on the 1st
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=14,
    )

    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-01"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 13
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_time_limits_create2():
    # We don't have 14 dates of data for an 8-day cohort:
    with pytest.raises(
        ValueError,
        match=(
            "You said you wanted 8 dates of enrollment, "
            + "and need data from the 13th day after enrollment. "
            + "For that, you need to wait until we have data for 2019-01-21."
        ),
    ):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date="2019-01-01",
            last_date_full_data="2019-01-14",
            analysis_start_days=0,
            analysis_length_dates=14,
            num_dates_enrollment=8,
        )

    # We don't have 15 full dates of data for any users
    with pytest.raises(AssertionError):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date="2019-01-01",
            last_date_full_data="2019-01-14",
            analysis_start_days=0,
            analysis_length_dates=15,
        )


def test_time_limits_create3():
    # For the 8-day cohort We have enough data for a 7 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=7,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_time_limits_create4():
    # Or a 2 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=2,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 1
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-09"


def test_time_limits_create5():
    # But not an 8 day window
    with pytest.raises(
        ValueError,
        match=(
            "You said you wanted 8 dates of enrollment, "
            + "and need data from the 7th day after enrollment. "
            + "For that, you need to wait until we have data for 2019-01-15."
        ),
    ):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date="2019-01-01",
            last_date_full_data="2019-01-14",
            analysis_start_days=0,
            analysis_length_dates=8,
            num_dates_enrollment=8,
        )


def test_time_limits_create6():
    # Of course the flexi-experiment has data for a 1 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=0,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-14"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 0
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_time_limits_create7():
    # If the analysis starts later, so does the data source
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        analysis_start_days=7,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-07"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 7
    assert tl.analysis_windows[0].end == 7
    assert tl.first_date_data_required == "2019-01-08"
    assert tl.last_date_data_required == "2019-01-14"


def test_ts_time_limits_create1():
    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        time_series_period="daily",
        num_dates_enrollment=8,
    )

    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 7
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 0
    assert tl.analysis_windows[6].start == 6
    assert tl.analysis_windows[6].end == 6
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_ts_time_limits_create2():
    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-14",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_ts_time_limits_create3():
    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-01-15",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-01-14"


def test_ts_time_limits_create4():
    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-02-15",
        time_series_period="28_day",
        num_dates_enrollment=8,
    )

    assert tl.first_enrollment_date == "2019-01-01"
    assert tl.last_enrollment_date == "2019-01-08"
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 27
    assert tl.first_date_data_required == "2019-01-01"
    assert tl.last_date_data_required == "2019-02-04"


def test_ts_time_limits_create_not_enough_data():
    with pytest.raises(ValueError, match="Insufficient data"):
        TimeLimits.for_ts(
            first_enrollment_date="2019-01-01",
            last_date_full_data="2019-01-13",
            time_series_period="weekly",
            num_dates_enrollment=8,
        )


def test_time_limits_has_right_date_in_error_message():
    msg_re = r"until we have data for 2020-03-30."
    with pytest.raises(ValueError, match=msg_re):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date="2020-03-03",
            last_date_full_data="2020-03-23",
            analysis_start_days=0,
            analysis_length_dates=21,
            num_dates_enrollment=8,
        )


def test_analysis_window_validates_start():
    AnalysisWindow(0, 1)
    AnalysisWindow(-2, -1)
    with pytest.raises(AssertionError):
        AnalysisWindow(-1, 1)


def test_analysis_window_validates_end():
    AnalysisWindow(5, 5)
    with pytest.raises(AssertionError):
        AnalysisWindow(5, 4)


def test_query_not_detectably_malformed():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.NORMANDY,
        sample_size=None,
    )

    sql_lint(enrollments_sql)
    assert "sample_id < None" not in enrollments_sql

    metrics_sql = exp.build_metrics_query(
        metric_list=[],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)


def test_megaquery_not_detectably_malformed():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl, enrollments_query_type=EnrollmentsQueryType.NORMANDY
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=desktop_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)


def test_segments_megaquery_not_detectably_malformed():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        segment_list=desktop_segments,
        enrollments_query_type=EnrollmentsQueryType.NORMANDY,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=desktop_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)


def test_app_id_propagates():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    sds = SegmentDataSource(
        name="cool_data_source",
        from_expr="`moz-fx-data-shared-prod`.{dataset}.cool_table",
        default_dataset="org_mozilla_firefox",
    )

    segment = Segment(
        name="cool_segment",
        select_expr="COUNT(*)",
        data_source=sds,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        segment_list=[segment],
        enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=fenix_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    assert "org_mozilla_firefox" not in enrollments_sql
    assert "my_cool_app" in enrollments_sql

    sql_lint(metrics_sql)


def test_query_not_detectably_malformed_fenix_fallback():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK,
        sample_size=10,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=[],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)


def test_firefox_ios_app_id_propagation():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    sds = SegmentDataSource(
        name="cool_data_source",
        from_expr="`moz-fx-data-shared-prod`.{dataset}.cool_table",
        default_dataset="org_mozilla_ios_firefox",
    )

    segment = Segment(
        name="cool_segment",
        select_expr="COUNT(*)",
        data_source=sds,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        segment_list=[segment],
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=firefox_ios_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    assert "org_mozilla_ios_firefox" not in enrollments_sql
    assert "my_cool_app" in enrollments_sql

    sql_lint(metrics_sql)


def test_firefox_klar_app_id_propagation():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    sds = SegmentDataSource(
        name="cool_data_source",
        from_expr="`moz-fx-data-shared-prod`.{dataset}.cool_table",
        default_dataset="org_mozilla_klar",
    )

    segment = Segment(
        name="cool_segment",
        select_expr="COUNT(*)",
        data_source=sds,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        segment_list=[segment],
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=klar_android_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    assert "org_mozilla_klar" not in enrollments_sql
    assert "my_cool_app" in enrollments_sql

    sql_lint(metrics_sql)


def test_firefox_ios_klar_app_id_propagation():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    sds = SegmentDataSource(
        name="cool_data_source",
        from_expr="`moz-fx-data-shared-prod`.{dataset}.cool_table",
        default_dataset="org_mozilla_ios_klar",
    )

    segment = Segment(
        name="cool_segment",
        select_expr="COUNT(*)",
        data_source=sds,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl,
        segment_list=[segment],
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=klar_ios_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    assert "org_mozilla_ios_klar" not in enrollments_sql
    assert "my_cool_app" in enrollments_sql

    sql_lint(metrics_sql)


def test_enrollments_query_cirrus():
    exp = Experiment("slug", "2019-01-01", 8, app_id="monitor_cirrus")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollment_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.CIRRUS,
    )

    sql_lint(enrollment_sql)

    assert "exposures" in enrollment_sql
    assert 'mozfun.map.get_key(e.extra, "user_id")' in enrollment_sql
    assert "cirrus_events" in enrollment_sql
    assert 'mozfun.map.get_key(event.extra, "user_id")' in enrollment_sql


def test_exposure_query():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollment_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
    )

    sql_lint(enrollment_sql)

    assert "exposures" in enrollment_sql


def test_exposure_signal_query():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollment_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
        exposure_signal=ExposureSignal(
            name="exposures",
            data_source=ConfigLoader.get_data_source("baseline", "fenix"),
            select_expr="metrics.counter.events_total_uri_count > 0",
            friendly_name="URI visited exposure",
            description="Exposed when URI visited",
        ),
    )

    sql_lint(enrollment_sql)

    assert "exposures" in enrollment_sql
    assert "metrics.counter.events_total_uri_count > 0" in enrollment_sql


def test_exposure_signal_query_custom_windows():
    exp = Experiment("slug", "2019-01-01", 8, app_id="my_cool_app")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollment_sql = exp.build_enrollments_query(
        time_limits=tl,
        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
        exposure_signal=ExposureSignal(
            name="exposures",
            data_source=ConfigLoader.get_data_source("baseline", "fenix"),
            select_expr="metrics.counter.events_total_uri_count > 0",
            friendly_name="URI visited exposure",
            description="Exposed when URI visited",
            window_start=1,
            window_end=3,
        ),
    )

    sql_lint(enrollment_sql)

    assert "exposures" in enrollment_sql
    assert "metrics.counter.events_total_uri_count > 0" in enrollment_sql
    assert "DATE_ADD('2019-01-01', INTERVAL 1 DAY)" in enrollment_sql
    assert "DATE_ADD('2019-01-01', INTERVAL 3 DAY)" in enrollment_sql


def test_metrics_query_based_on_exposure():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl, enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=fenix_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
        analysis_basis=AnalysisBasis.EXPOSURES,
    )

    sql_lint(metrics_sql)

    assert "e.exposure_date" in metrics_sql


def test_metrics_query_with_exposure_signal_custom_windows():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl, enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=fenix_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
        analysis_basis=AnalysisBasis.EXPOSURES,
        exposure_signal=ExposureSignal(
            name="exposures",
            data_source=ConfigLoader.get_data_source("baseline", "fenix"),
            select_expr="metrics.counter.events_total_uri_count > 0",
            friendly_name="URI visited exposure",
            description="Exposed when URI visited",
            window_start=1,
            window_end=3,
        ),
    )

    sql_lint(metrics_sql)

    assert "DATE_ADD('2019-01-01', INTERVAL 1 DAY)" in metrics_sql
    assert "DATE_ADD('2019-01-01', INTERVAL 3 DAY)" in metrics_sql


def test_metrics_query_with_exposure_signal():
    exp = Experiment("slug", "2019-01-01", 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    enrollments_sql = exp.build_enrollments_query(
        time_limits=tl, enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK
    )

    sql_lint(enrollments_sql)

    metrics_sql = exp.build_metrics_query(
        metric_list=fenix_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
        analysis_basis=AnalysisBasis.EXPOSURES,
        exposure_signal=ExposureSignal(
            name="exposures",
            data_source=ConfigLoader.get_data_source("baseline", "fenix"),
            select_expr="metrics.counter.events_total_uri_count > 0",
            friendly_name="URI visited exposure",
            description="Exposed when URI visited",
        ),
    )

    sql_lint(metrics_sql)

    assert "DATE_ADD('2019-01-01', INTERVAL 0 DAY)" in metrics_sql
    assert "DATE_ADD('2019-01-08', INTERVAL 0 DAY)" in metrics_sql


def test_resolve_metric_slugs():
    exp = Experiment("slug", "2019-01-01", 8, "org_mozilla_fenix", "fenix")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    metrics_sql = exp.build_metrics_query(
        metric_list=["baseline_ping_count"],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    assert "baseline" in metrics_sql


def test_resolve_invalid_metric_slugs():
    exp = Experiment("slug", "2019-01-01", 8, "org_mozilla_fenix", "fenix")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(
        Exception, match="Could not find definition for metric not_exist"
    ):
        exp.build_metrics_query(
            metric_list=["not_exist"],
            time_limits=tl,
            enrollments_table="enrollments",
        )


def test_resolve_invalid_app_name():
    exp = Experiment("slug", "2019-01-01", 8, "unknown", "unknown")

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(ApplicationNotFound, match="Could not find application unknown"):
        exp.build_metrics_query(
            metric_list=["baseline_ping_count"],
            time_limits=tl,
            enrollments_table="enrollments",
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

    test_exp = Experiment(experiment_slug="test_exp", start_date="2022-01-01")

    metric_sql = test_exp.build_metrics_query(
        metric_list=[test_metric], time_limits=tl, enrollments_table="test_table"
    )

    assert "None" not in metric_sql
