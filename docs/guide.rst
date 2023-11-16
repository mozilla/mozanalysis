=====
Guide
=====

Basic experiment: get the data & crunch the stats
=================================================

Let's start by analysing a straightforward pref-flip experiment on the desktop browser.

If we're using a Colab notebook, we begin by installing the latest version of :mod:`mozanalysis` into the notebook. It's a good idea to specify the specific version (`from pypi <https://pypi.org/project/mozanalysis/>`_), for reproducibility::

    !pip install mozanalysis=='{current_version}'

We take the per-notebook daily trudge::

    from google.colab import auth
    auth.authenticate_user()
    print('Authenticated')

Then we import the necessary classes for getting the data, and for analysing the data, and for interacting with BigQuery::

    import mozanalysis.bayesian_stats.binary as mabsbin
    from mozanalysis.experiment import Experiment
    from mozanalysis.bq import BigQueryContext


And get a :class:`mozanalysis.bq.BigQueryContext` (a client, and some config)
::

    bq_context = BigQueryContext(
        dataset_id='your_dataset_id',  # e.g. mine's 'flawrence'
        project_id=...,  # Defaults to moz-fx-data-bq-data-science
    )

If you do not have a dataset, you will need to `create one <https://cloud.google.com/bigquery/docs/datasets#create-dataset>`_. Mozanalysis will save data into this dataset - if you want to access them directly (i.e. not through mozanalysis), they live at ``project_id.dataset_id.table_name`` , where ``table_name`` will be printed by mozanalysis when it saves/retrieves the data.

To bill queries to a project other than ``moz-fx-data-bq-data-science``, pass the ``project_id`` as an argument when initializing your :class:`mozanalysis.bq.BigQueryContext`.

For querying data, the internal approach of :mod:`mozanalysis` is to start by obtaining a list of who was enrolled in what branch, when. Then we try to quantify what happened to each client: for a given analysis window (a specified period of time defined with respect to the client's enrollment date), we seek to obtain a value for each client for each metric. We end up with a results (pandas) DataFrame with one row per client and one column per metric.


We start by instantiating our :class:`mozanalysis.experiment.Experiment` object::

    exp = Experiment(
        experiment_slug='pref-fingerprinting-protections-retention-study-release-70',
        start_date='2019-10-29',
        num_dates_enrollment=8,
        app_name="firefox_desktop"
    )

``start_date`` is the ``submission_date`` of the first enrollment (``submission_date`` is in UTC). If you intended to study one week's worth of enrollments, then set ``num_dates_enrollment=8``: Normandy experiments typically go live in the evening UTC-time, so 8 days of data is a better approximation than 7.


We now gather a list of who was enrolled in what branch and when, and try to quantify what happened to each client. In many cases, the metrics in which you're interested will already be in a metrics library, `metric-hub <https://github.com/mozilla/metric-hub>`_. If not, then you can define your own---see :class:`mozanalysis.metrics.Metric` for examples---and ideally submit a PR to add them to metric-hub for the next experiment. To load a Metric from metric-hub, for example::

    from mozanalysis.config import ConfigLoader
    active_hours = ConfigLoader.get_metric(slug="active_hours", app_name="firefox_desktop")

In this example, we'll compute four metrics from metric-hub:

* active hours
* uri count
* ad clicks
* search count

As it happens, the first three metrics all come from the ``clients_daily`` dataset, whereas "search count" comes from ``search_clients_daily``. These details are taken care of in the `metric-hub definitions <https://github.com/mozilla/metric-hub/tree/main/definitions>`_ so that we don't have to think about them here.

A metric must be computed over some `analysis window`, a period of time defined with respect to the enrollment date. We could use :meth:`mozanalysis.experiment.Experiment.get_single_window_data()` to compute our metrics over a specific analysis window. But here, let's create time series data: let's have an analysis window for each of the first three weeks of the experiment, and measure the data for each of these analysis windows::

    ts_res = exp.get_time_series_data(
        bq_context=bq_context,
        metric_list=[
            active_hours,
            uri_count,
            ad_clicks,
            search_count,
        ],
        last_date_full_data='2019-11-28',
        time_series_period='weekly'
    )

The first two arguments to :meth:`mozanalysis.experiment.Experiment.get_time_series_data()` should be clear by this point. ``last_date_full_data`` is the last date for which we want to use data. For a currently-running experiment, it would typically be yesterday's date (we have incomplete data for incomplete days!).

Metrics are pulled in from `metric-hub <https://github.com/mozilla/metric-hub>`_ based on the provided metric slugs.

``time_series_period`` can be ``'daily'``, ``'weekly'`` or ``'28_day'``. A ``'weekly'`` time series neatly sidesteps/masks weekly seasonality issues: most of the experiment subjects will enroll within a day of the experiment launching - typically a Tuesday, leading to ``'daily'`` time series reflecting a non-uniform convolution of the metrics' weekly seasonalities with the uneven enrollment numbers across the week.

:meth:`mozanalysis.experiment.Experiment.get_time_series_data()` returns a :class:`mozanalysis.experiment.TimeSeriesResult` object, which can return DataFrames keyed by the start of their analysis windows (measured in days after enrollment)::

    >>> ts_res.keys()
    [0, 7, 14]

If RAM permits, we can dump all the results into a ``dict`` of DataFrames keyed by the start of their analysis windows::

    res = dict(ts_res.items(bq_context))

Each value in ``res`` is a pandas DataFrame in "the standard format", with one row per enrolled client and one column per metric.

Otherwise you might want to load one analysis window at a time, by calling ``ts_res.get(bq_context, analysis_window_start)`` for each analysis window in ``ts_res.keys()``, processing the resulting DataFrame, then discarding the DataFrame from RAM before moving onto the next analysis window.

Here are the columns of each result DataFrame::

    >>> res[7].columns
    Index(['branch', 'enrollment_date', 'num_enrollment_events', 'active_hours',
           'uri_count', 'clients_daily_has_contradictory_branch',
           'clients_daily_has_non_enrolled_data', 'ad_clicks', 'search_count'],
          dtype='object')

The 'branch' column contains the client's branch::

    >>> res[7].branch.unique()
    array(['treatment', 'control'], dtype=object)

And we can do the usual pandas DataFrame things - e.g. calculate the mean active hours per branch::

    >>> res[7].groupby('branch').active_hours.mean()
    branch
    Cohort_1    6.246536
    Cohort_2    6.719880
    Cohort_3    6.468948
    Name: active_hours, dtype: float64

Suppose we want to see whether the user had any active hours in their second week in the experiment. This information can be calculated from the ``active_hours`` metric - we add this as a column to the results pandas DataFrame, then use :mod:`mozanalysis.bayesian_stats.binary` to analyse this data::

    res[7]['active_hours_gt_0'] = res[7]['active_hours'] > 0

    retention_week_2 = mabsbin.compare_branches(res[7], 'active_hours_gt_0', ref_branch_label='Cohort_1')

Like most of the stats in :mod:`mozanalysis`, :func:`mozanalysis.bayesian_stats.binary.compare_branches()` accepts a pandas DataFrame in "the standard format" and returns credible (or confidence) intervals for various quantities. It expects the reference branch to be named 'control'; since this experiment used non-standard branch naming, we need to tell it that the control branch is named 'Cohort_1'. The function returns credible intervals (CIs) for the fraction of active users in each branch.::

    >>> retention_week_2['individual']
    {'Cohort_1':
         0.005    0.733865
         0.025    0.734265
         0.5      0.735536
         0.975    0.736803
         0.995    0.737201
         mean     0.735535
         dtype: float64,
     'Cohort_2':
         0.005    0.732368
         0.025    0.732769
         0.5      0.734041
         0.975    0.735312
         0.995    0.735710
         mean     0.734041
         dtype: float64,
     'Cohort_3':
         0.005    0.732289
         0.025    0.732690
         0.5      0.733962
         0.975    0.735232
         0.995    0.735630
         mean     0.733962
         dtype: float64}

(output re-wrapped for clarity)

For example, we can see that the fraction of users in Cohort_2 with >0 active hours in week 2 has an expectation value of 0.734, with a 95% CI of (0.7328, 0.7353).

And the function also returns credible intervals for the uplift in this quantity for each branch with respect to a reference branch::

    >>> retention_week_2['comparative']
    {'Cohort_3':
        rel_uplift    0.005   -0.005222
                      0.025   -0.004568
                      0.5     -0.002173
                      0.975    0.000277
                      0.995    0.001056
                      exp     -0.002166
        abs_uplift    0.005   -0.003850
                      0.025   -0.003365
                      0.5     -0.001598
                      0.975    0.000204
                      0.995    0.000774
                      exp     -0.001594
        max_abs_diff  0.95     0.003092
        prob_win      NaN      0.041300
        dtype: float64,
     'Cohort_2':
        rel_uplift    0.005   -0.005215
                      0.025   -0.004502
                      0.5     -0.002065
                      0.975    0.000359
                      0.995    0.001048
                      exp     -0.002066
        abs_uplift    0.005   -0.003840
                      0.025   -0.003314
                      0.5     -0.001520
                      0.975    0.000264
                      0.995    0.000769
                      exp     -0.001520
        max_abs_diff  0.95     0.003043
        prob_win      NaN      0.046800
        dtype: float64}

(output re-wrapped for clarity)

``rel_uplift`` contains quantities related to the relative uplift of a branch with respect to the reference branch (as given by ``ref_branch_label``); for example, assuming a uniform prior, there is a 95% probability that Cohort_3 had between 0.457% fewer and 0.028% more users with >0 active hours in the second week, compared to Cohort_1. ``abs_uplift`` refers to the absolute uplifts, and ``prob_win`` gives the probability that the branch is better than the reference branch.

Since :mod:`mozanalysis` is designed around this "standard format", you can pass any of the values in ``res`` to any of the statistics functions, as long as the statistics are suited to the column's type (i.e. binary vs real-valued data)::

    import mozanalysis.bayesian_stats.binary as mabsbin
    retention_week_2 = mabsbin.compare_branches(res[7], 'active_hours_gt_0')

    import mozanalysis.frequentist_stats.bootstrap as mafsboot
    boot_uri_week_1 = mafsboot.compare_branches(res[0], 'uri_count', threshold_quantile=0.9999)

    import mozanalysis.bayesian_stats.survival_func as mabssf
    sf_search_week_2 = mabssf.compare_branches(res[7], 'search_count')

:mod:`dscontrib.flawrence.plot_experiments` has some (shaky) support for visualising stats over time series experiment results.


Get the data: cookbook
=============================

Time series (of analysis windows)
---------------------------------
Condensing the above example for simpler copying and pasting::

    !pip install mozanalysis=='{current_version}'

    from google.colab import auth
    auth.authenticate_user()
    print('Authenticated')

    import mozanalysis.bayesian_stats.binary as mabsbin
    from mozanalysis.experiment import Experiment
    from mozanalysis.bq import BigQueryContext
    from mozanalysis.config import ConfigLoader

    bq_context = BigQueryContext(dataset_id='your_dataset_id')

    active_hours = ConfigLoader.get_metric(slug="active_hours", app_name="firefox_desktop")
    uri_count = ConfigLoader.get_metric(slug="uri_count", app_name="firefox_desktop")
    ad_clicks = ConfigLoader.get_metric(slug="ad_clicks", app_name="firefox_desktop")
    search_count = ConfigLoader.get_metric(slug="search_count", app_name="firefox_desktop")
    
    ts_res = exp.get_time_series_data(
        bq_context=bq_context,
        metric_list=[
            active_hours,
            uri_count,
            ad_clicks,
            search_count,
        ],
        last_date_full_data='2019-11-28',
        time_series_period='weekly'
    )

    res = dict(ts_res.items(bq_context))

One analysis window
-------------------

If we're only interested in users' (say) second week in the experiment, then we don't need to get a full time series.
::

    !pip install mozanalysis=='{current_version}'

    from google.colab import auth
    auth.authenticate_user()
    print('Authenticated')

    import mozanalysis.bayesian_stats.binary as mabsbin
    from mozanalysis.experiment import Experiment
    from mozanalysis.bq import BigQueryContext
    from mozanalysis.config import ConfigLoader

    bq_context = BigQueryContext(dataset_id='your_dataset_id')
    
    active_hours = ConfigLoader.get_metric(slug="active_hours", app_name="firefox_desktop")

    res = exp.get_single_window_data(
        bq_context=bq_context,
        metric_list=[
            active_hours,
        ],
        last_date_full_data='2019-01-07',
        analysis_start_days=7,
        analysis_length_days=7
    )

``last_date_full_data`` is less important for :meth:`mozanalysis.experiment.Experiment.get_single_window_data` than for :meth:`mozanalysis.experiment.Experiment.get_time_series_data`: while ``last_date_full_data`` determines the length of the time series, here it simply sanity checks that the specified analysis window doesn't stretch into the future for any enrolled users.


Crunch the stats
================

Each stats technique has a module in :mod:`mozanalysis.bayesian_stats` or :mod:`mozanalysis.frequentist_stats`, and a function ``compare_branches()``; for example :func:`mozanalysis.bayesian_stats.binary.compare_branches`. This function accepts a pandas DataFrame in "the standard format", and must be passed the name of the column containing the metric to be studied.
::

    import mozanalysis.bayesian_stats.binary as mabsbin
    import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsboot
    import mozanalysis.bayesian_stats.survival_func as mabssf
    import mozanalysis.frequentist_stats.bootstrap as mafsboot

    res_from_ts[7]['active_hours_gt_0'] = res_from_ts[7].active_hours_gt_0 > 0
    mabsbin.compare_branches(res_from_ts[7], 'active_hours_gt_0')
    mabsbin.compare_branches(res_from_ts[7], 'active_hours_gt_0', ref_branch_label='Cohort_1')

    gpcd_res['active_hours_gt_0'] = gpcd_res.active_hours_gt_0 > 0
    mabsbin.compare_branches(gpcd_res, 'active_hours_gt_0')

    mafsboot.compare_branches(gpcd_res, 'active_hours', threshold_quantile=0.9999)

    sf_search_week_2 = mabssf.compare_branches(gpcd_res, 'search_count')
