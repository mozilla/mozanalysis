=====
Guide
=====

Basic experiment: get the data & crunch the stats
=================================================

Let's start by analysing a straightforward pref-flip experiment on the desktop browser.

If we're using Databricks, we begin by installing the latest version of :mod:`mozanalysis` into the notebook (we don't install for the whole cluster, because then upgrades require restarting the entire cluster)::

    dbutils.library.installPyPI("mozanalysis", "[current version]")

Then we import the necessary classes for getting the data, and for analysing the data::

    import mozanalysis.metrics.desktop as mmd
    import mozanalysis.bayesian_stats.binary as mabsbin
    from mozanalysis.experiment import Experiment

For querying data, the general approach of :mod:`mozanalysis` is to start by obtaining a list of who was enrolled in what branch, when. This list is the ``enrollments`` Spark DataFrame; it has one row per client. Then we try to quantify what happened to each client: for a given analysis window (a specified period of time defined with respect to the client's experiment results.date), we seek to obtain a value for each client for each metric. So we end up with a results (pandas) DataFrame with one row per client and one column per metric.


We start by instantiating our :class:`mozanalysis.experiment.Experiment` object::

    exp = Experiment(
        experiment_slug='pref-flip-defaultoncookierestrictions-1506704',
        start_date='20181217',
        num_dates_enrollment=8
    )

``start_date`` is the ``submission_date_s3`` of the first enrollment (``submission_date_s3`` is in UTC). If you intended to study one week's worth of enrollments, then set ``num_dates_enrollment=8``: Normandy experiments typically go live in the evening UTC-time, so 8 days of data is a better approximation than 7.

The :class:`mozanalysis.experiment.Experiment` instance has a method that gives us the ``enrollments`` Spark DataFrame. It's a good idea to cache it - we may use it multiple times::

    enrollments = exp.get_enrollments(spark).cache()

``enrollments`` has one row per enrolled client, and three columns::

    >>> enrollments.columns
    ['client_id', 'branch', 'enrollment_date']

Having obtained our list of who was enrolled in what branch and when, we now try to quantify what happened to each client. In many cases, the metrics in which you're interested will already be in a metrics library, a submodule of :mod:`mozanalysis.metrics`. If not, then you can define your own - see :meth:`mozanalysis.metrics.Metric` for examples - and ideally submit a PR to add them to the library for the next experiment. In this example, we'll compute four metrics:

* :const:`mozanalysis.metrics.desktop.active_hours`
* uri count
* ad clicks
* search count

As it happens, the first three metrics all come from the ``clients_daily`` dataset, whereas "search count" comes from ``search_clients_daily``. These details are taken care of in the :class:`mozanalysis.metrics.Metric` definitions so that we don't have to think about them here.

A metric must be computed over some `analysis window`, a period of time defined with respect to the enrollment date. We could use :meth:`mozanalysis.experiment.Experiment.get_per_client_data()` to compute our metrics over a specific analysis window. But here, let's create time series data: let's have an analysis window for each of the first three weeks of the experiment, and measure the data for each of these analysis windows::

    res = exp.get_time_series_data(
        enrollments=enrollments,
        metric_list=[
            mmd.active_hours,
            mmd.uri_count,
            mmd.ad_clicks,
            mmd.search_count,
        ],
        last_date_full_data='20190107',
        time_series_period='weekly'
    )

The first two arguments to :meth:`mozanalysis.experiment.Experiment.get_time_series_data()` should be clear by this point. ``last_date_full_data`` is the last date for which we want to use data. For a currently-running experiment, it would typically be yesterday's date (we have incomplete data for incomplete days!). Here I chose a date that gives us two weeks of data for the last eligible enrollees, who enrolled on '20181224' (yes, this experiment ran over the holidays...).

``time_series_period`` can be ``'daily'`` or ``'weekly'``. A ``'weekly'`` time series neatly sidesteps/masks weekly seasonality issues: most of the experiment subjects will enroll within a day of the experiment launching - typically a Tuesday, leading to ``'daily'`` time series reflecting a non-uniform convolution of the metrics' weekly seasonalities with the uneven enrollment numbers across the week.

:meth:`mozanalysis.experiment.Experiment.get_time_series_data()` returns a ``dict`` keyed by the start of the analysis window (measured in days after enrollment)::

    >>> res.keys()
    dict_keys([0, 7])

Each value is a pandas DataFrame in "the standard format", with one row per client from the ``enrollments`` Spark DataFrame, and one column per metric::

    >>> res[7].columns
     Index(['branch', 'enrollment_date', 'active_hours', 'uri_count',
       'ad_clicks', 'clients_daily_has_contradictory_branch',
       'clients_daily_has_non_enrolled_data', 'search_count',
       'search_clients_daily_has_contradictory_branch',
       'search_clients_daily_has_non_enrolled_data'],
      dtype='object')

The 'branch' column contains the client's branch::

    >>> res[7].branch.unique()
    array(['Cohort_1', 'Cohort_3', 'Cohort_2'], dtype=object)

And we can do the usual pandas DataFrame things - e.g. calculate the mean active hours per branch::

    >>> res[7].groupby('branch').active_hours.mean()
    branch
    Cohort_1    6.246536
    Cohort_2    6.719880
    Cohort_3    6.468948
    Name: active_hours, dtype: float64

Suppose we want to see whether the user had any active hours in their second week in the experiment. This information can be calculated from the ``mmd.active_hours`` metric - we add this as a column to the results pandas DataFrame, then use :mod:`mozanalysis.bayesian_stats.binary` to analyse this data::

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

    dbutils.library.installPyPI("mozanalysis", "[current version]")

    import mozanalysis.metrics.desktop as mmd
    from mozanalysis.experiment import Experiment

    exp = Experiment(
        experiment_slug='pref-flip-defaultoncookierestrictions-1506704',
        start_date='20181217',
        num_dates_enrollment=8
    )

    enrollments = exp.get_enrollments(spark).cache()

    res = exp.get_time_series_data(
        enrollments=enrollments,
        metric_list=[
            mmd.active_hours,
        ],
        last_date_full_data='20190107',
        time_series_period='weekly'
    )


One analysis window
-------------------

If we're only interested in users' (say) second week in the experiment, then we don't need to get a full time series.
::

    dbutils.library.installPyPI("mozanalysis", "[current version]")

    import mozanalysis.metrics.desktop as mmd
    from mozanalysis.experiment import Experiment

    exp = Experiment(
        experiment_slug='pref-flip-defaultoncookierestrictions-1506704',
        start_date='20181217',
        num_dates_enrollment=8
    )

    enrollments = exp.get_enrollments(spark).cache()

    res = exp.get_per_client_data(
        enrollments=enrollments,
        metric_list=[
            mmd.active_hours,
        ],
        last_date_full_data='20190107',
        analysis_start_days=7,
        analysis_length_days=7
    )

``last_date_full_data`` is less important for :meth:`mozanalysis.experiment.Experiment.get_per_client_data` than for :meth:`mozanalysis.experiment.Experiment.get_time_series_data`: while ``last_date_full_data`` determines the length of the time series, here it simply sanity checks that the specified analysis window doesn't stretch into the future for any enrolled users.


Crunch the stats
================

Each stats technique has a module in :mod:`mozanalysis.bayesian_stats` or :mod:`mozanalysis.frequentist_stats`, and a function ``compare_branches()``; for example :func:`mozanalysis.bayesian_stats.binary.compare_branches`. This function accepts a pandas DataFrame in "the standard format", and must be passed the name of the column containing the metric to be studied.
::

    import mozanalysis.bayesian_stats.binary as mabsbin
    import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsboot
    import mozanalysis.bayesian_stats.survival_func as mabssf
    import mozanalysis.frequentist_stats.bootstrap as mafsboot

    ts_res[7]['active_hours_gt_0'] = ts_res[7].active_hours_gt_0 > 0
    mabsbin.compare_branches(ts_res[7], 'active_hours_gt_0')
    mabsbin.compare_branches(ts_res[7], 'active_hours_gt_0', ref_branch_label='Cohort_1')

    gpcd_res['active_hours_gt_0'] = gpcd_res.active_hours_gt_0 > 0
    mabsbin.compare_branches(gpcd_res, 'active_hours_gt_0')

    mafsboot.compare_branches(gpcd_res, 'active_hours', threshold_quantile=0.9999)

    sf_search_week_2 = mabssf.compare_branches(gpcd_res, 'search_count')
