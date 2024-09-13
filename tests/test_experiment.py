from textwrap import dedent

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
from metric_config_parser import AnalysisUnit

from mozanalysis.config import ApplicationNotFound, ConfigLoader
from mozanalysis.experiment import (
    AnalysisWindow,
    EnrollmentsQueryType,
    Experiment,
    IncompatibleAnalysisUnit,
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


@pytest.mark.parametrize(
    "analysis_unit", [AnalysisUnit.CLIENT, AnalysisUnit.PROFILE_GROUP]
)
def test_query_not_detectably_malformed(analysis_unit: AnalysisUnit):
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=analysis_unit)

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

    if analysis_unit == AnalysisUnit.CLIENT:
        assert "client_id" in enrollments_sql
    elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
        assert "profile_group_id" in enrollments_sql

    metrics_sql = exp.build_metrics_query(
        metric_list=[],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    if analysis_unit == AnalysisUnit.CLIENT:
        assert "client_id" in metrics_sql
    elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
        assert "profile_group_id" in metrics_sql


@pytest.mark.parametrize(
    "analysis_unit", [AnalysisUnit.CLIENT, AnalysisUnit.PROFILE_GROUP]
)
def test_megaquery_not_detectably_malformed(analysis_unit: AnalysisUnit):
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=analysis_unit)

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

    if analysis_unit == AnalysisUnit.CLIENT:
        assert "client_id" in enrollments_sql
    elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
        assert "profile_group_id" in enrollments_sql

    metrics_sql = exp.build_metrics_query(
        metric_list=desktop_metrics,
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    if analysis_unit == AnalysisUnit.CLIENT:
        assert "client_id" in metrics_sql
    elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
        assert "profile_group_id" in metrics_sql


@pytest.mark.parametrize(
    "analysis_unit", [AnalysisUnit.CLIENT, AnalysisUnit.PROFILE_GROUP]
)
def test_segments_megaquery_not_detectably_malformed(
    analysis_unit: AnalysisUnit,
):
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=analysis_unit)

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


def test_enrollments_query_explicit_client_id():
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

    expected = """
    WITH raw_enrollments AS (
SELECT
    e.client_id,
    `mozfun.map.get_key`(e.event_map_values, 'branch')
        AS branch,
    MIN(e.submission_date) AS enrollment_date,
    COUNT(e.submission_date) AS num_enrollment_events
FROM
    `moz-fx-data-shared-prod.telemetry.events` e
WHERE
    e.event_category = 'normandy'
    AND e.event_method = 'enroll'
    AND e.submission_date
        BETWEEN '2019-01-01' AND '2019-01-08'
    AND e.event_string_value = 'slug'
    AND e.sample_id < 100
GROUP BY e.client_id, branch
    ),
    segmented_enrollments AS (
SELECT
    raw_enrollments.*,

FROM raw_enrollments

),
    exposures AS (
SELECT
    e.client_id,
    e.branch,
    min(e.submission_date) AS exposure_date,
    COUNT(e.submission_date) AS num_exposure_events
FROM raw_enrollments re
LEFT JOIN (
    SELECT
        client_id,
        `mozfun.map.get_key`(event_map_values, 'branchSlug') AS branch,
        submission_date
    FROM
        `moz-fx-data-shared-prod.telemetry.events`
    WHERE
        event_category = 'normandy'
        AND (event_method = 'exposure' OR event_method = 'expose')
        AND submission_date
            BETWEEN '2019-01-01' AND '2019-01-08'
        AND event_string_value = 'slug'
) e
ON re.client_id = e.client_id AND
    re.branch = e.branch AND
    e.submission_date >= re.enrollment_date
GROUP BY e.client_id, e.branch
    )

    SELECT
        se.*,
        e.* EXCEPT (client_id, branch)
    FROM segmented_enrollments se
    LEFT JOIN exposures e
    USING (client_id, branch)
"""

    assert dedent(enrollments_sql) == expected

    metrics_sql = exp.build_metrics_query(
        metric_list=[
            metric for metric in desktop_metrics if metric.name == "active_hours"
        ],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)


def test_metrics_query_explicit_client_id():
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
        metric_list=[
            metric for metric in desktop_metrics if metric.name == "active_hours"
        ],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    expected = """
WITH analysis_windows AS (
    (SELECT 0 AS analysis_window_start, 6 AS analysis_window_end)
UNION ALL
(SELECT 7 AS analysis_window_start, 13 AS analysis_window_end)
UNION ALL
(SELECT 14 AS analysis_window_start, 20 AS analysis_window_end)
UNION ALL
(SELECT 21 AS analysis_window_start, 27 AS analysis_window_end)
UNION ALL
(SELECT 28 AS analysis_window_start, 34 AS analysis_window_end)
UNION ALL
(SELECT 35 AS analysis_window_start, 41 AS analysis_window_end)
UNION ALL
(SELECT 42 AS analysis_window_start, 48 AS analysis_window_end)
),
raw_enrollments AS (
    -- needed by "exposures" sub query
    SELECT
        e.*,
        aw.*
    FROM `enrollments` e
    CROSS JOIN analysis_windows aw
),
exposures AS (
        SELECT
            *
        FROM raw_enrollments e
    ),
enrollments AS (
    SELECT
        e.* EXCEPT (exposure_date, num_exposure_events),
        x.exposure_date,
        x.num_exposure_events
    FROM exposures x
        RIGHT JOIN raw_enrollments e
        USING (client_id, branch)
)
SELECT
    enrollments.*,
    ds_0.active_hours
FROM enrollments
    LEFT JOIN (
    SELECT
    e.client_id,
    e.branch,
    e.analysis_window_start,
    e.analysis_window_end,
    e.num_exposure_events,
    e.exposure_date,
    COALESCE(SUM(active_hours_sum), 0) AS active_hours
FROM enrollments e
    LEFT JOIN mozdata.telemetry.clients_daily ds
        ON ds.client_id = e.client_id
        AND ds.submission_date BETWEEN '2019-01-01' AND '2019-02-25'
        AND ds.submission_date BETWEEN
            DATE_ADD(e.enrollment_date, interval e.analysis_window_start day)
            AND DATE_ADD(e.enrollment_date, interval e.analysis_window_end day)

GROUP BY
    e.client_id,
    e.branch,
    e.num_exposure_events,
    e.exposure_date,
    e.analysis_window_start,
    e.analysis_window_end
    ) ds_0 USING (client_id, branch, analysis_window_start, analysis_window_end)"""

    assert expected == dedent(metrics_sql.rstrip())


def test_enrollments_query_explicit_group_id():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

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

    expected = """
    WITH raw_enrollments AS (
SELECT
    e.profile_group_id,
    `mozfun.map.get_key`(e.event_map_values, 'branch')
        AS branch,
    MIN(e.submission_date) AS enrollment_date,
    COUNT(e.submission_date) AS num_enrollment_events
FROM
    `moz-fx-data-shared-prod.telemetry.events` e
WHERE
    e.event_category = 'normandy'
    AND e.event_method = 'enroll'
    AND e.submission_date
        BETWEEN '2019-01-01' AND '2019-01-08'
    AND e.event_string_value = 'slug'
    AND e.sample_id < 100
GROUP BY e.profile_group_id, branch
    ),
    segmented_enrollments AS (
SELECT
    raw_enrollments.*,

FROM raw_enrollments

),
    exposures AS (
SELECT
    e.profile_group_id,
    e.branch,
    min(e.submission_date) AS exposure_date,
    COUNT(e.submission_date) AS num_exposure_events
FROM raw_enrollments re
LEFT JOIN (
    SELECT
        profile_group_id,
        `mozfun.map.get_key`(event_map_values, 'branchSlug') AS branch,
        submission_date
    FROM
        `moz-fx-data-shared-prod.telemetry.events`
    WHERE
        event_category = 'normandy'
        AND (event_method = 'exposure' OR event_method = 'expose')
        AND submission_date
            BETWEEN '2019-01-01' AND '2019-01-08'
        AND event_string_value = 'slug'
) e
ON re.profile_group_id = e.profile_group_id AND
    re.branch = e.branch AND
    e.submission_date >= re.enrollment_date
GROUP BY e.profile_group_id, e.branch
    )

    SELECT
        se.*,
        e.* EXCEPT (profile_group_id, branch)
    FROM segmented_enrollments se
    LEFT JOIN exposures e
    USING (profile_group_id, branch)
"""

    assert dedent(enrollments_sql) == expected


def test_metrics_query_explicit_group_id():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

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
        metric_list=[
            metric for metric in desktop_metrics if metric.name == "active_hours"
        ],
        time_limits=tl,
        enrollments_table="enrollments",
    )

    sql_lint(metrics_sql)

    expected = """
WITH analysis_windows AS (
    (SELECT 0 AS analysis_window_start, 6 AS analysis_window_end)
UNION ALL
(SELECT 7 AS analysis_window_start, 13 AS analysis_window_end)
UNION ALL
(SELECT 14 AS analysis_window_start, 20 AS analysis_window_end)
UNION ALL
(SELECT 21 AS analysis_window_start, 27 AS analysis_window_end)
UNION ALL
(SELECT 28 AS analysis_window_start, 34 AS analysis_window_end)
UNION ALL
(SELECT 35 AS analysis_window_start, 41 AS analysis_window_end)
UNION ALL
(SELECT 42 AS analysis_window_start, 48 AS analysis_window_end)
),
raw_enrollments AS (
    -- needed by "exposures" sub query
    SELECT
        e.*,
        aw.*
    FROM `enrollments` e
    CROSS JOIN analysis_windows aw
),
exposures AS (
        SELECT
            *
        FROM raw_enrollments e
    ),
enrollments AS (
    SELECT
        e.* EXCEPT (exposure_date, num_exposure_events),
        x.exposure_date,
        x.num_exposure_events
    FROM exposures x
        RIGHT JOIN raw_enrollments e
        USING (profile_group_id, branch)
)
SELECT
    enrollments.*,
    ds_0.active_hours
FROM enrollments
    LEFT JOIN (
    SELECT
    e.profile_group_id,
    e.branch,
    e.analysis_window_start,
    e.analysis_window_end,
    e.num_exposure_events,
    e.exposure_date,
    COALESCE(SUM(active_hours_sum), 0) AS active_hours
FROM enrollments e
    LEFT JOIN mozdata.telemetry.clients_daily ds
        ON ds.profile_group_id = e.profile_group_id
        AND ds.submission_date BETWEEN '2019-01-01' AND '2019-02-25'
        AND ds.submission_date BETWEEN
            DATE_ADD(e.enrollment_date, interval e.analysis_window_start day)
            AND DATE_ADD(e.enrollment_date, interval e.analysis_window_end day)

GROUP BY
    e.profile_group_id,
    e.branch,
    e.num_exposure_events,
    e.exposure_date,
    e.analysis_window_start,
    e.analysis_window_end
    ) ds_0 USING (profile_group_id, branch, analysis_window_start, analysis_window_end)"""  # noqa: E501

    assert expected == dedent(metrics_sql.rstrip())


def test_glean_group_id_incompatible():
    exp = Experiment(
        "slug",
        "2019-01-01",
        8,
        analysis_unit=AnalysisUnit.PROFILE_GROUP,
        app_id="test_app",
    )

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp.build_enrollments_query(
            time_limits=tl, enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT
        )


def test_glean_group_id_incompatible_exposures():
    exp = Experiment(
        "slug",
        "2019-01-01",
        8,
        analysis_unit=AnalysisUnit.PROFILE_GROUP,
        app_id="test_app",
    )

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp._build_exposure_query(
            time_limits=tl, exposure_query_type=EnrollmentsQueryType.GLEAN_EVENT
        )


def test_glean_missing_app_id():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(
        ValueError, match="App ID must be defined for building Glean enrollments query"
    ):
        exp.build_enrollments_query(
            time_limits=tl, enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT
        )


def test_glean_exposures_missing_app_id():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(
        ValueError, match="App ID must be defined for building Glean exposures query"
    ):
        exp._build_exposure_query(
            time_limits=tl, exposure_query_type=EnrollmentsQueryType.GLEAN_EVENT
        )


def test_cirrus_group_id_incompatible():
    exp = Experiment(
        "slug",
        "2019-01-01",
        8,
        analysis_unit=AnalysisUnit.PROFILE_GROUP,
        app_id="test_app",
    )

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp.build_enrollments_query(
            time_limits=tl, enrollments_query_type=EnrollmentsQueryType.CIRRUS
        )


def test_cirrus_group_id_incompatible_exposures():
    exp = Experiment(
        "slug",
        "2019-01-01",
        8,
        analysis_unit=AnalysisUnit.PROFILE_GROUP,
        app_id="test_app",
    )

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp._build_exposure_query(
            time_limits=tl, exposure_query_type=EnrollmentsQueryType.CIRRUS
        )


def test_cirrus_missing_app_id():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(
        ValueError, match="App ID must be defined for building Cirrus enrollments query"
    ):
        exp.build_enrollments_query(
            time_limits=tl, enrollments_query_type=EnrollmentsQueryType.CIRRUS
        )


def test_cirrus_missing_app_id_exposures():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(
        ValueError, match="App ID must be defined for building Cirrus exposures query"
    ):
        exp._build_exposure_query(
            time_limits=tl, exposure_query_type=EnrollmentsQueryType.CIRRUS
        )


def test_fenix_group_id_incompatible():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp.build_enrollments_query(
            time_limits=tl, enrollments_query_type=EnrollmentsQueryType.FENIX_FALLBACK
        )


def test_fenix_group_id_incompatible_exposures():
    exp = Experiment("slug", "2019-01-01", 8, analysis_unit=AnalysisUnit.PROFILE_GROUP)

    tl = TimeLimits.for_ts(
        first_enrollment_date="2019-01-01",
        last_date_full_data="2019-03-01",
        time_series_period="weekly",
        num_dates_enrollment=8,
    )

    with pytest.raises(IncompatibleAnalysisUnit):
        exp._build_exposure_query(
            time_limits=tl, exposure_query_type=EnrollmentsQueryType.FENIX_FALLBACK
        )
