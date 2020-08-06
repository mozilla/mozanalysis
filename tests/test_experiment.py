import pytest
from cheap_lint import sql_lint

import mozanalysis.metrics.desktop as mad
import mozanalysis.segments.desktop as msd
from mozanalysis.experiment import TimeLimits, AnalysisWindow, Experiment


def test_time_limits_validates():
    # Mainly check that the validation is running at all
    # No need to specify the same checks twice(?)
    with pytest.raises(TypeError):
        TimeLimits()

    with pytest.raises(AssertionError):
        TimeLimits(
            first_enrollment_date='2019-01-05',
            last_enrollment_date='2019-01-05',
            analysis_windows=(AnalysisWindow(1, 1),),
            first_date_data_required='2019-01-01',  # Before enrollments
            last_date_data_required='2019-01-01',
        )


def test_time_limits_create1():
    # When we have complete data for 2019-01-14...
    # ...We have 14 dates of data for those who enrolled on the 1st
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        analysis_start_days=0,
        analysis_length_dates=14,
    )

    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-01'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 13
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_time_limits_create2():
    # We don't have 14 dates of data for an 8-day cohort:
    with pytest.raises(ValueError):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date='2019-01-01',
            last_date_full_data='2019-01-14',
            analysis_start_days=0,
            analysis_length_dates=14,
            num_dates_enrollment=8,
        )

    # We don't have 15 full dates of data for any users
    with pytest.raises(AssertionError):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date='2019-01-01',
            last_date_full_data='2019-01-14',
            analysis_start_days=0,
            analysis_length_dates=15,
        )


def test_time_limits_create3():
    # For the 8-day cohort We have enough data for a 7 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        analysis_start_days=0,
        analysis_length_dates=7,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-08'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_time_limits_create4():
    # Or a 2 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        analysis_start_days=0,
        analysis_length_dates=2,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-08'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 1
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-09'


def test_time_limits_create5():
    # But not an 8 day window
    with pytest.raises(ValueError):
        TimeLimits.for_single_analysis_window(
            first_enrollment_date='2019-01-01',
            last_date_full_data='2019-01-14',
            analysis_start_days=0,
            analysis_length_dates=8,
            num_dates_enrollment=8,
        )


def test_time_limits_create6():
    # Of course the flexi-experiment has data for a 1 day window
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        analysis_start_days=0,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-14'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 0
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_time_limits_create7():
    # If the analysis starts later, so does the data source
    tl = TimeLimits.for_single_analysis_window(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        analysis_start_days=7,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-07'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 7
    assert tl.analysis_windows[0].end == 7
    assert tl.first_date_data_required == '2019-01-08'
    assert tl.last_date_data_required == '2019-01-14'


def test_ts_time_limits_create1():
    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        time_series_period='daily',
        num_dates_enrollment=8
    )

    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-08'
    assert len(tl.analysis_windows) == 7
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 0
    assert tl.analysis_windows[6].start == 6
    assert tl.analysis_windows[6].end == 6
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_ts_time_limits_create2():
    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-14',
        time_series_period='weekly',
        num_dates_enrollment=8
    )

    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-08'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_ts_time_limits_create3():
    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-01-15',
        time_series_period='weekly',
        num_dates_enrollment=8
    )

    assert tl.first_enrollment_date == '2019-01-01'
    assert tl.last_enrollment_date == '2019-01-08'
    assert len(tl.analysis_windows) == 1
    assert tl.analysis_windows[0].start == 0
    assert tl.analysis_windows[0].end == 6
    assert tl.first_date_data_required == '2019-01-01'
    assert tl.last_date_data_required == '2019-01-14'


def test_ts_time_limits_create_not_enough_data():
    with pytest.raises(ValueError):
        TimeLimits.for_ts(
            first_enrollment_date='2019-01-01',
            last_date_full_data='2019-01-13',
            time_series_period='weekly',
            num_dates_enrollment=8
        )


def test_time_limits_has_right_date_in_error_message():
    msg_re = r'until we have data for 2020-03-30.'
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
    with pytest.raises(AssertionError):
        AnalysisWindow(-1, 1)


def test_analysis_window_validates_end():
    AnalysisWindow(5, 5)
    with pytest.raises(AssertionError):
        AnalysisWindow(5, 4)


def test_query_not_detectably_malformed():
    exp = Experiment('slug', '2019-01-01', 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-03-01',
        time_series_period='weekly',
        num_dates_enrollment=8
    )

    sql = exp.build_query(
        metric_list=[],
        time_limits=tl,
        enrollments_query_type='normandy',
    )

    sql_lint(sql)


def test_megaquery_not_detectably_malformed():
    exp = Experiment('slug', '2019-01-01', 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-03-01',
        time_series_period='weekly',
        num_dates_enrollment=8
    )

    sql = exp.build_query(
        metric_list=[m for m in mad.__dict__.values() if isinstance(m, mad.Metric)],
        time_limits=tl,
        enrollments_query_type='normandy',
    )

    sql_lint(sql)


def test_segments_megaquery_not_detectably_malformed():
    exp = Experiment('slug', '2019-01-01', 8)

    tl = TimeLimits.for_ts(
        first_enrollment_date='2019-01-01',
        last_date_full_data='2019-03-01',
        time_series_period='weekly',
        num_dates_enrollment=8
    )

    sql = exp.build_query(
        metric_list=[m for m in mad.__dict__.values() if isinstance(m, mad.Metric)],
        segment_list=[s for s in msd.__dict__.values() if isinstance(s, msd.Segment)],
        time_limits=tl,
        enrollments_query_type='normandy',
    )

    sql_lint(sql)
