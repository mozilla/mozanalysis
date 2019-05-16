import pyspark.sql.functions as F
import pytest

from mozanalysis.experiment import Experiment
from mozanalysis.utils import add_days


def test_get_last_enrollment_date():
    exp = Experiment('a-stub', '20190101')
    exp_8d = Experiment('experiment-with-8-day-cohort', '20190101', 8)

    # When we have complete data for 20190114...
    the_fourteenth = '20190114'

    # ...We have 14 dates of data for those who enrolled on the 1st
    exp._get_last_enrollment_date(the_fourteenth, 14) == '20190101'

    # We don't have 14 dates of data for the 8-day cohort:
    with pytest.raises(ValueError):
        exp_8d._get_last_enrollment_date(the_fourteenth, 14)

    # We don't have 15 full dates of data for any users
    with pytest.raises(ValueError):
        exp._get_last_enrollment_date(the_fourteenth, 15)

    # And we certainly don't have 15 full dates for the 8-day cohort:
    with pytest.raises(ValueError):
        exp_8d._get_last_enrollment_date(the_fourteenth, 15)

    # For the 8-day cohort We have enough data for a 7 day window
    exp_8d._get_last_enrollment_date(the_fourteenth, 7) == '20190108'

    # Or a 2 day window
    exp_8d._get_last_enrollment_date(the_fourteenth, 2) == '20190108'

    # But not an 8 day window
    with pytest.raises(ValueError):
        exp_8d._get_last_enrollment_date(the_fourteenth, 8)

    # Of course the flexi-experiment has data for a 1 day window
    exp._get_last_enrollment_date(the_fourteenth, 1) == '20190114'


def test_get_last_data_date1():
    exp = Experiment('a-stub', '20190101')

    # When we have complete data for 20190114...
    the_fourteenth = '20190114'

    # When we don't specify num_days_enrollment we'll use all the data
    assert exp._get_last_data_date(the_fourteenth, 14) == the_fourteenth
    assert exp._get_last_data_date(the_fourteenth, 10) == the_fourteenth
    assert exp._get_last_data_date(the_fourteenth, 1) == the_fourteenth
    assert exp._get_last_data_date(the_fourteenth, 5) == the_fourteenth

    # But we don't have 15 full dates of data for any users
    with pytest.raises(ValueError):
        exp._get_last_data_date(the_fourteenth, 15)


def test_get_last_data_date2():
    exp_8d = Experiment('experiment-with-8-day-cohort', '20190101', 8)

    the_fourteenth = '20190114'

    with pytest.raises(ValueError):
        assert exp_8d._get_last_data_date(the_fourteenth, 14) == the_fourteenth

    with pytest.raises(ValueError):
        assert exp_8d._get_last_data_date(the_fourteenth, 10) == the_fourteenth

    # If we only need 1 date of data then the final enrollment is fixed:
    assert exp_8d._get_last_data_date(the_fourteenth, 1) == '20190108'
    assert exp_8d._get_last_data_date('20190131', 1) == '20190108'
    assert exp_8d._get_last_data_date('20201231', 1) == '20190108'

    # And it's fixed to the date of the final enrollment
    assert exp_8d._get_last_data_date(the_fourteenth, 1) \
        == '20190108' \
        == exp_8d._get_last_enrollment_date(the_fourteenth, 1)

    assert exp_8d._get_last_data_date(the_fourteenth, 5) == '20190112'


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


def test_filter_data_source_for_analysis_window(spark):
    start_date = '20190101'
    exp_8d = Experiment('experiment-with-8-day-cohort', start_date, 8)
    data_source = _get_data_source(spark)

    end_date = '20190114'

    # Are the fixtures sufficiently complicated that we're actually testing
    # things?
    assert _simple_return_agg_date(F.min, data_source) < start_date
    assert _simple_return_agg_date(F.max, data_source) > end_date

    filtered_ds = exp_8d.filter_data_source_for_analysis_window(
        data_source, end_date, 0, 3
    )

    assert _simple_return_agg_date(F.min, filtered_ds) == start_date
    assert _simple_return_agg_date(F.max, filtered_ds) == '20190110'

    filtered_ds_2 = exp_8d.filter_data_source_for_analysis_window(
        data_source, end_date, 2, 3
    )

    assert _simple_return_agg_date(F.min, filtered_ds_2) == add_days(start_date, 2)
    assert _simple_return_agg_date(F.max, filtered_ds_2) == '20190112'


def _get_enrollment_view(spark, slug):
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


def test_filter_enrollments_for_analysis_window(spark, monkeypatch):
    exp = Experiment('a-stub', '20190101')
    _mock_exp(monkeypatch, exp)
    enrollments = exp.get_enrollments(spark)
    assert enrollments.count() == 4

    # With final data collected on '20190114', we have 7 dates of data
    # for 'cccc' enrolled on '20190108' but not for 'dddd' enrolled on
    # '20190109'.
    fe = exp.filter_enrollments_for_analysis_window(enrollments, '20190114', 7)
    assert fe.count() == 3


def _mock_exp(monkeypatch, exp):
    monkeypatch.setattr(
        exp, '_get_enrollments_view_normandy',
        lambda spark: _get_enrollment_view(spark, exp.experiment_slug)
    )
    monkeypatch.setattr(
        exp, '_get_enrollments_view_addon',
        lambda spark, addon_version: _get_enrollment_view(spark, exp.experiment_slug)
    )


def test_get_enrollments(spark, monkeypatch):
    # Experiment = _mock_exp(monkeypatch)

    exp = Experiment('a-stub', '20190101')
    _mock_exp(monkeypatch, exp)
    assert exp.get_enrollments(spark).count() == 4

    exp2 = Experiment('a-stub2', '20190102')
    _mock_exp(monkeypatch, exp2)
    enrl2 = exp2.get_enrollments(spark, study_type='addon')
    assert enrl2.count() == 2
    assert enrl2.select(F.min(enrl2.enrollment_date).alias('b')).first(
        )['b'] == '20190108'

    exp_8d = Experiment('experiment-with-8-day-cohort', '20190101', 8)
    _mock_exp(monkeypatch, exp_8d)
    enrl_8d = exp_8d.get_enrollments(spark)
    assert enrl_8d.count() == 3
    assert enrl_8d.select(F.max(enrl_8d.enrollment_date).alias('b')).first(
        )['b'] == '20190108'


def test_get_enrollments_debug_dupes(spark, monkeypatch):
    exp = Experiment('a-stub', '20190101')
    _mock_exp(monkeypatch, exp)

    enrl = exp.get_enrollments(spark)
    assert 'num_events' not in enrl.columns

    enrl2 = exp.get_enrollments(spark, debug_dupes=True)
    assert 'num_events' in enrl2.columns

    penrl2 = enrl2.toPandas()
    assert (penrl2['num_events'] == 1).all()


def test_get_per_client_data_doesnt_crash(spark):
    exp = Experiment('a-stub', '20190101', 8)
    enrollments = _get_enrollment_view(spark, exp.experiment_slug)
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
            # but missed by `filter_data_source_for_analysis_window`
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
    # i.e. check the join, not just _filter_data_source_for_analysis_window
    bob_badtiming = res.filter(res.client_id == 'bob-badtiming')
    assert bob_badtiming.count() == 1
    assert bob_badtiming.first()['some_value'] == 0
    # Check that _filter_data_source_for_analysis_window didn't do the
    # heavy lifting above
    fds = exp.filter_data_source_for_analysis_window(data_source, '20190114', 1, 3)
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
