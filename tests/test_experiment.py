import pyspark.sql.functions as F
import pytest

from pyspark.sql.utils import AnalysisException

from mozanalysis.experiment import Experiment, TimeLimits
from mozanalysis.utils import add_days


def test_time_limits_validates():
    # Mainly check that the validation is running at all
    # No need to specify the same checks twice(?)
    with pytest.raises(TypeError):
        TimeLimits()

    with pytest.raises(AssertionError):
        TimeLimits(
            first_enrollment_date='20190105',
            last_enrollment_date='20190105',
            analysis_window_start=1,
            analysis_window_end=1,
            analysis_window_length_dates=1,
            first_date_data_required='20190101',  # Before enrollments
            last_date_data_required='20190101',
        )


def test_time_limits_create1():
    # When we have complete data for 20190114...
    # ...We have 14 dates of data for those who enrolled on the 1st
    tl = TimeLimits.create(
        first_enrollment_date='20190101',
        last_date_full_data='20190114',
        analysis_start_days=0,
        analysis_length_dates=14,
    )

    assert tl.first_enrollment_date == '20190101'
    assert tl.last_enrollment_date == '20190101'
    assert tl.analysis_window_start == 0
    assert tl.analysis_window_end == 13
    assert tl.analysis_window_length_dates == 14
    assert tl.first_date_data_required == '20190101'
    assert tl.last_date_data_required == '20190114'


def test_time_limits_create2():
    # We don't have 14 dates of data for an 8-day cohort:
    with pytest.raises(ValueError):
        TimeLimits.create(
            first_enrollment_date='20190101',
            last_date_full_data='20190114',
            analysis_start_days=0,
            analysis_length_dates=14,
            num_dates_enrollment=8,
        )

    # We don't have 15 full dates of data for any users
    with pytest.raises(AssertionError):
        TimeLimits.create(
            first_enrollment_date='20190101',
            last_date_full_data='20190114',
            analysis_start_days=0,
            analysis_length_dates=15,
        )


def test_time_limits_create3():
    # For the 8-day cohort We have enough data for a 7 day window
    tl = TimeLimits.create(
        first_enrollment_date='20190101',
        last_date_full_data='20190114',
        analysis_start_days=0,
        analysis_length_dates=7,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == '20190101'
    assert tl.last_enrollment_date == '20190108'
    assert tl.analysis_window_start == 0
    assert tl.analysis_window_end == 6
    assert tl.analysis_window_length_dates == 7
    assert tl.first_date_data_required == '20190101'
    assert tl.last_date_data_required == '20190114'


def test_time_limits_create4():
    # Or a 2 day window
    tl = TimeLimits.create(
        first_enrollment_date='20190101',
        last_date_full_data='20190114',
        analysis_start_days=0,
        analysis_length_dates=2,
        num_dates_enrollment=8,
    )
    assert tl.first_enrollment_date == '20190101'
    assert tl.last_enrollment_date == '20190108'
    assert tl.analysis_window_start == 0
    assert tl.analysis_window_end == 1
    assert tl.analysis_window_length_dates == 2
    assert tl.first_date_data_required == '20190101'
    assert tl.last_date_data_required == '20190109'


def test_time_limits_create5():
    # But not an 8 day window
    with pytest.raises(ValueError):
        TimeLimits.create(
            first_enrollment_date='20190101',
            last_date_full_data='20190114',
            analysis_start_days=0,
            analysis_length_dates=8,
            num_dates_enrollment=8,
        )


def test_time_limits_create6():
    # Of course the flexi-experiment has data for a 1 day window
    tl = TimeLimits.create(
        first_enrollment_date='20190101',
        last_date_full_data='20190114',
        analysis_start_days=0,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == '20190101'
    assert tl.last_enrollment_date == '20190114'
    assert tl.analysis_window_start == 0
    assert tl.analysis_window_end == 0
    assert tl.analysis_window_length_dates == 1
    assert tl.first_date_data_required == '20190101'
    assert tl.last_date_data_required == '20190114'


def test_time_limits_create7():
    # If the analysis starts later, so does the data source
    tl = TimeLimits.create(
        first_enrollment_date='20190101',
        last_date_full_data='20190114',
        analysis_start_days=7,
        analysis_length_dates=1,
    )
    assert tl.first_enrollment_date == '20190101'
    assert tl.last_enrollment_date == '20190107'
    assert tl.analysis_window_start == 7
    assert tl.analysis_window_end == 7
    assert tl.analysis_window_length_dates == 1
    assert tl.first_date_data_required == '20190108'
    assert tl.last_date_data_required == '20190114'


def _get_data_source(spark):
    clients_branches = [
        ('aaaa', 'control'),
        ('bbbb', 'test'),
    ]
    dates = [add_days('20181215', i) for i in range(32)]

    data_rows = [
        [
            client, submission_date_s3, {'a-stub': branch}, 1.
        ]
        for client, branch in clients_branches
        for submission_date_s3 in dates
    ]

    return spark.createDataFrame(
        data_rows,
        [
            "client_id",
            "submission_date_s3",
            "experiments",
            "constant_one",
        ],
    )


def _simple_return_agg_date(agg_fn, data_source):
    return data_source.select(
        agg_fn(data_source.submission_date_s3).alias('b')
    ).first()['b']


def test_process_data_source(spark):
    start_date = '20190101'
    exp_8d = Experiment('experiment-with-8-day-cohort', start_date, 8)
    data_source = _get_data_source(spark)

    end_date = '20190114'

    # Are the fixtures sufficiently complicated that we're actually testing
    # things?
    assert _simple_return_agg_date(F.min, data_source) < start_date
    assert _simple_return_agg_date(F.max, data_source) > end_date

    tl_03 = TimeLimits.create(
        first_enrollment_date=exp_8d.start_date,
        last_date_full_data=end_date,
        analysis_start_days=0,
        analysis_length_dates=3,
        num_dates_enrollment=exp_8d.num_dates_enrollment
    )
    assert tl_03.first_date_data_required == start_date
    assert tl_03.last_date_data_required == '20190110'

    proc_ds = exp_8d._process_data_source(data_source, tl_03)

    assert _simple_return_agg_date(F.min, proc_ds) == tl_03.first_date_data_required
    assert _simple_return_agg_date(F.max, proc_ds) == tl_03.last_date_data_required

    tl_23 = TimeLimits.create(
        first_enrollment_date=exp_8d.start_date,
        last_date_full_data=end_date,
        analysis_start_days=2,
        analysis_length_dates=3,
        num_dates_enrollment=exp_8d.num_dates_enrollment
    )
    assert tl_23.first_date_data_required == add_days(start_date, 2)
    assert tl_23.last_date_data_required == '20190112'

    p_ds_2 = exp_8d._process_data_source(data_source, tl_23)

    assert _simple_return_agg_date(F.min, p_ds_2) == tl_23.first_date_data_required
    assert _simple_return_agg_date(F.max, p_ds_2) == tl_23.last_date_data_required

    assert proc_ds.select(F.col('data_source.client_id'))
    with pytest.raises(AnalysisException):
        assert data_source.select(F.col('data_source.client_id'))


def _get_enrollment_view(slug):
    def inner(spark):
        # `slug` is supplied so we reuse this fixture
        # with multiple slugs
        data_rows = [
            ['aaaa', slug, 'control', '20190101'],
            ['bbbb', slug, 'test', '20190101'],
            ['cccc', slug, 'control', '20190108'],
            ['dddd', slug, 'test', '20190109'],
            ['eeee', 'no', 'control', '20190101'],
        ]

        return spark.createDataFrame(
            data_rows,
            [
                "client_id",
                "experiment_slug",
                "branch",
                "enrollment_date",
            ],
        )
    return inner


def test_process_enrollments(spark):
    exp = Experiment('a-stub', '20190101')
    enrollments = exp.get_enrollments(
        spark,
        _get_enrollment_view(slug="a-stub")
    )
    assert enrollments.count() == 4

    # With final data collected on '20190114', we have 7 dates of data
    # for 'cccc' enrolled on '20190108' but not for 'dddd' enrolled on
    # '20190109'.
    tl = TimeLimits.create(
        first_enrollment_date=exp.start_date,
        last_date_full_data='20190114',
        analysis_start_days=0,
        analysis_length_dates=7,
        num_dates_enrollment=exp.num_dates_enrollment
    )
    assert tl.last_enrollment_date == '20190108'
    assert tl.analysis_window_end == 6

    fe = exp._process_enrollments(enrollments, tl)
    assert fe.count() == 3

    fe = exp._process_enrollments(enrollments.alias('main_summary'), tl)
    assert fe.select(F.col('enrollments.enrollment_date'))
    with pytest.raises(AnalysisException):
        assert fe.select(F.col('main_summary.enrollment_date'))


def test_get_enrollments(spark):
    exp = Experiment('a-stub', '20190101')
    view_method = _get_enrollment_view("a-stub")
    assert exp.get_enrollments(spark, view_method).count() == 4

    exp2 = Experiment('a-stub2', '20190102')
    view_method2 = _get_enrollment_view("a-stub2")
    enrl2 = exp2.get_enrollments(spark, study_type=view_method2)
    assert enrl2.count() == 2
    assert enrl2.select(F.min(enrl2.enrollment_date).alias('b')).first(
        )['b'] == '20190108'

    exp_8d = Experiment('experiment-with-8-day-cohort', '20190101', 8)
    view_method_8d = _get_enrollment_view("experiment-with-8-day-cohort")
    enrl_8d = exp_8d.get_enrollments(spark, view_method_8d)
    assert enrl_8d.count() == 3
    assert enrl_8d.select(F.max(enrl_8d.enrollment_date).alias('b')).first(
        )['b'] == '20190108'


def test_get_enrollments_debug_dupes(spark):
    exp = Experiment('a-stub', '20190101')
    view_method = _get_enrollment_view("a-stub")

    enrl = exp.get_enrollments(spark, view_method)
    assert 'num_events' not in enrl.columns

    enrl2 = exp.get_enrollments(spark, view_method, debug_dupes=True)
    assert 'num_events' in enrl2.columns

    penrl2 = enrl2.toPandas()
    assert (penrl2['num_events'] == 1).all()


def test_get_per_client_data_doesnt_crash(spark):
    exp = Experiment('a-stub', '20190101', 8)
    enrollments = _get_enrollment_view(exp.experiment_slug)(spark)
    data_source = _get_data_source(spark)

    exp.get_per_client_data(
        enrollments,
        data_source,
        [
            F.sum(data_source.constant_one).alias('something_meaningless'),
        ],
        '20190114',
        0,
        3
    )


def test_get_per_client_data_join(spark):
    exp = Experiment('a-stub', '20190101')

    enrollments = spark.createDataFrame(
        [
            ['aaaa', 'control', '20190101'],
            ['bbbb', 'test', '20190101'],
            ['cccc', 'control', '20190108'],
            ['dddd', 'test', '20190109'],
            ['annie-nodata', 'control', '20190101'],
            ['bob-badtiming', 'test', '20190102'],
            ['carol-gooddata', 'test', '20190101'],
            ['derek-lateisok', 'control', '20190110'],
        ],
        [
            "client_id",
            "branch",
            "enrollment_date",
        ],
    )

    ex_d = {'a-stub': 'fake-branch-lifes-too-short'}
    data_source = spark.createDataFrame(
        [
            # bob-badtiming only has data before/after analysis window
            # but missed by `process_data_source`
            ['bob-badtiming', '20190102', ex_d, 1],
            ['bob-badtiming', '20190106', ex_d, 2],
            # carol-gooddata has data on two days (including a dupe day)
            ['carol-gooddata', '20190102', ex_d, 3],
            ['carol-gooddata', '20190102', ex_d, 2],
            ['carol-gooddata', '20190104', ex_d, 6],
            # derek-lateisok has data before and during the analysis window
            ['derek-lateisok', '20190110', ex_d, 1000],
            ['derek-lateisok', '20190111', ex_d, 1],
            # TODO: exercise the last condition on the join
        ],
        [
            "client_id",
            "submission_date_s3",
            "experiments",
            "some_value",
        ],
    )

    res = exp.get_per_client_data(
        enrollments,
        data_source,
        [
            F.coalesce(F.sum(data_source.some_value), F.lit(0)).alias('some_value'),
        ],
        '20190114',
        1,
        3,
        keep_client_id=True
    )

    # Check that the dataframe has the correct number of rows
    assert res.count() == enrollments.count()

    # Check that dataless enrollments are handled correctly
    annie_nodata = res.filter(res.client_id == 'annie-nodata')
    assert annie_nodata.count() == 1
    assert annie_nodata.first()['some_value'] == 0

    # Check that early and late data were ignored
    # i.e. check the join, not just _process_data_source
    bob_badtiming = res.filter(res.client_id == 'bob-badtiming')
    assert bob_badtiming.count() == 1
    assert bob_badtiming.first()['some_value'] == 0
    # Check that _process_data_source didn't do the
    # heavy lifting above
    time_limits = TimeLimits.create(
        exp.start_date, '20190114', 1, 3, exp.num_dates_enrollment
    )
    fds = exp._process_data_source(data_source, time_limits)
    assert fds.filter(
        fds.client_id == 'bob-badtiming'
    ).select(
        F.sum(fds.some_value).alias('agg_val')
    ).first()['agg_val'] == 3

    # Check that relevant data was included appropriately
    carol_gooddata = res.filter(res.client_id == 'carol-gooddata')
    assert carol_gooddata.count() == 1
    assert carol_gooddata.first()['some_value'] == 11

    derek_lateisok = res.filter(res.client_id == 'derek-lateisok')
    assert derek_lateisok.count() == 1
    assert derek_lateisok.first()['some_value'] == 1

    # Check that it still works for `data_source`s without an experiments map
    res2 = exp.get_per_client_data(
        enrollments,
        data_source.drop('experiments'),
        [
            F.coalesce(F.sum(data_source.some_value), F.lit(0)).alias('some_value'),
        ],
        '20190114',
        1,
        3,
        keep_client_id=True
    )

    assert res2.count() == enrollments.count()


def test_no_analysis_exception_when_shared_parent_dataframe(spark):
    # Check that we don't fall victim to
    # https://issues.apache.org/jira/browse/SPARK-10925
    df = spark.createDataFrame(
        [  # Just need the schema, really
            ['someone', '20190102', 'fake', 1],
        ],
        [
            "client_id",
            "submission_date_s3",
            "branch",
            "some_value",
        ]
    )

    enrollments = df.groupby(
        'client_id', 'branch'
    ).agg(
        F.min('submission_date_s3').alias('enrollment_date')
    )

    exp = Experiment('a-stub', '20180101')

    exp.get_per_client_data(
        enrollments,
        df,
        [
            F.max(F.col('some_value'))
        ],
        last_date_full_data='20190522',
        analysis_start_days=28,
        analysis_length_days=7
    )
