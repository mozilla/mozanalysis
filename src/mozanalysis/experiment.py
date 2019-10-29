# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr

from functools import reduce
from pyspark.sql import functions as F

from mozanalysis.utils import add_days, date_sub


@attr.s(frozen=True, slots=True)
class Experiment(object):
    """Get DataFrames of experiment data; store experiment metadata.

    The methods here query data in a way compatible with the following
    principles, which are important for experiment analysis:

    * The population of clients in each branch must have the same
      properties, aside from the intervention itself and its
      consequences; i.e. there must be no underlying bias in the
      branch populations.
    * We must measure the same thing for each client, to minimize the
      variance associated with our measurement.

    So that our analyses follow these abstract principles, we follow
    these rules:

    * Start with a list of all clients who enrolled.
    * We can filter this list of clients only based on information known
      to us at or before the time that they enrolled, because later
      information might be causally connected to the intervention.
    * For any given metric, every client gets a non-null value; we don't
      implicitly ignore anyone, even if they churned and stopped
      sending data.
    * Typically if an enrolled client no longer qualifies for enrollment,
      we'll still want to include their data in the analysis, unless
      we're explicitly using stats methods that handle censored data.
    * We define a "analysis window" with respect to clients'
      enrollment dates. Each metric only uses data collected inside
      this analysis window. We can only analyze data for a client
      if we have data covering their entire analysis window.


    Example usage::

        from mozanalysis.metrics.desktop import active_hours, uri_count


        experiment = Experiment(
            experiment_slug='pref-flip-defaultoncookierestrictions-1506704',
            start_date='20181215',
            num_dates_enrollment=8
        )
        enrollments = experiment.get_enrollments(spark)

        res = experiment.get_per_client_data(
            enrollments,
            [
                active_hours,
                uri_count
            ],
            last_date_full_data='20190107',
            analysis_start_days=0,
            analysis_length_days=7
        )

        # Pull data into a pandas df, ready for running stats
        pres = res.toPandas()

    Args:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '20190101'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many dates
            of enrollments. If ``None`` then use the maximum number of dates
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        addon_version (str, optional): The version of the experiment addon.
            Some addon experiment slugs get reused - in those cases we need
            to filter on the addon version also.

    Attributes:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '20190101'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many days
            of enrollments. If ``None`` then use the maximum number of days
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        addon_version (str, optional): The version of the experiment addon.
            Some addon experiment slugs get reused - in those cases we need
            to filter on the addon version also.
    """

    experiment_slug = attr.ib()
    start_date = attr.ib()
    num_dates_enrollment = attr.ib(default=None)
    addon_version = attr.ib(default=None)

    def get_enrollments(
        self, spark, study_type='pref_flip', end_date=None, debug_dupes=False
    ):
        """Return a DataFrame of enrolled clients.

        This works for pref-flip and addon studies.

        The underlying queries are different for pref-flip vs addon
        studies, because as of 2019/04/02, branch information isn't
        reliably available in the ``events`` table for addon experiments:
        branch may be NULL for all enrollments. The enrollment
        information for them is most reliably available in
        ``telemetry_shield_study_parquet``. Once this issue is resolved,
        we will probably start using normandy events for all desktop
        studies.
        Ref: https://bugzilla.mozilla.org/show_bug.cgi?id=1536644

        Args:
            spark: The spark context.
            study_type (str): One of the following strings:

                * 'pref_flip'
                * 'addon'

                or a callable that accepts a spark context as an argument
                and returns a Spark DataFrame containing all enrollment events
                ever conducted using that method, with columns ``client_id``,
                ``experiment_slug``, ``branch``, ``enrollment_date``,
                and ``addon_version`` if it's relevant.

            end_date (str, optional): Ignore enrollments after this
                date: for faster queries on stale experiments. If you
                set ``num_dates_enrollment`` then do not set this; at best
                it would be redundant, at worst it's contradictory.

            debug_dupes (bool, optional): Include a column ``num_events``
                giving the number of enrollment events associated with
                the ``client_id`` and ``branch``.

        Returns:
            A Spark DataFrame of enrollment data. One row per
            enrollment. Columns:

                * client_id (str)
                * enrollment_date (str): e.g. '20190329'
                * branch (str)
                * num_events (int, optional)
        """
        if callable(study_type):
            enrollments = study_type(spark)
        elif study_type == 'pref_flip':
            enrollments = self._get_enrollments_view_normandy(spark)
        elif study_type == 'addon':
            enrollments = self._get_enrollments_view_addon(spark)
        # elif study_type == 'glean':
        #     raise NotImplementedError
        else:
            raise ValueError("Unrecognized study_type: {}".format(study_type))

        enrollments = enrollments.filter(
            enrollments.enrollment_date >= self.start_date
        ).filter(
            enrollments.experiment_slug == self.experiment_slug
        )

        if self.addon_version:
            if "addon_version" not in enrollments.columns:
                raise ValueError(
                    ("Experiment constructed with an addon_version but your  "
                     "study_type (%s) is incompatible with addon versions."
                     ).format(study_type)
                )
            enrollments = enrollments.filter(
                enrollments.addon_version == self.addon_version
            ).drop(enrollments.addon_version)

        if self.num_dates_enrollment is not None:
            if end_date is not None:
                raise ValueError(
                    "Don't specify both 'end_date' and "
                    "'num_dates_enrollment'; you might contradict yourself."
                )
            enrollments = enrollments.filter(
                enrollments.enrollment_date <= add_days(
                    self.start_date, self.num_dates_enrollment - 1
                )
            )
        elif end_date is not None:
            enrollments = enrollments.filter(
                enrollments.enrollment_date <= end_date
            )

        # Deduplicate enrollment events. Optionally keep track of what
        # had to be deduplicated. Deduplicating a client who enrolls in
        # multiple branches is left as an exercise to the reader :|
        enrollments = enrollments.groupBy(
            enrollments.client_id, enrollments.branch
        ).agg(*(
            [F.min(enrollments.enrollment_date).alias('enrollment_date')]
            + (
                [F.count(enrollments.enrollment_date).alias('num_events')]
                if debug_dupes else []
            )
        ))

        enrollments.cache()

        return enrollments

    def get_per_client_data(
        self, enrollments, metric_list, last_date_full_data,
        analysis_start_days, analysis_length_days, keep_client_id=False
    ):
        """Return a DataFrame containing per-client metric values.

        Args:
            enrollments: A spark DataFrame of enrollments, like the one
                returned by ``self.get_enrollments()``.
            metric_list (list of mozanalysis.metric.Metric): The metrics
                to analyze.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            analysis_start_days (int): the start of the analysis window,
                measured in 'days since the client enrolled'. We ignore data
                collected outside this analysis window.
            analysis_length_days (int): the length of the analysis window,
                measured in days.
            keep_client_id (bool): Whether to return a ``client_id`` column.
                Defaults to False to reduce memory usage of the results.

        Returns:
            A spark DataFrame of experiment data. One row per ``client_id``.
            Some metadata columns, then one column per metric in
            ``metric_list``, and one column per sanity-check metric.
            Columns (not necessarily in order):

                * client_id (str, optional): Not necessary for
                  "happy path" analyses.
                * branch (str): The client's branch
                * other columns of ``enrollments``.
                * [metric 1]: The client's value for the first metric in
                  ``metric_list``.
                * ...
                * [metric n]: The client's value for the nth (final)
                  metric in ``metric_list``.
                * [sanity check 1]: The client's value for the first
                  sanity check metric for the first data source that
                  supports sanity checks.
                * ...
                * [sanity check n]: The client's value for the last
                  sanity check metric for the last data source that
                  supports sanity checks.

            This format - the schema plus there being one row per
            enrolled client, regardless of whether the client has data
            in ``data_source`` - was agreed upon by the DS team, and is the
            standard format for queried experimental data.
        """
        time_limits = TimeLimits.for_single_analysis_window(
            self.start_date, last_date_full_data, analysis_start_days,
            analysis_length_days, self.num_dates_enrollment
        )

        return self._get_per_client_data(
            enrollments, metric_list, time_limits, keep_client_id
        ).drop(
            'mozanalysis_analysis_window_start'
        ).drop(
            'mozanalysis_analysis_window_end'
        )

    def get_time_series_data(
        self, enrollments, metric_list, last_date_full_data,
        time_series_period='weekly', keep_client_id=False
    ):
        """Return a dict containing DataFrames with per-client metric values.

        Equivalent to looping over ``get_per_client_data`` with
        different analysis windows, calling ``.toPandas()``, and
        putting the resulting pandas DataFrames into a dictionary keyed
        by the day the analysis window starts; but this should be more
        efficient.

        Args:
            enrollments: A spark DataFrame of enrollments, like the one
                returned by ``self.get_enrollments()``.

            metric_list (list of mozanalysis.metric.Metric):
                The metrics to analyze.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            time_series_period ('daily' or 'weekly'): How long each
                analysis window should be.
            keep_client_id (bool): Whether to return a ``client_id`` column.
                Defaults to False to reduce memory usage of the results.

        Returns:
            A ``dict`` of data per analysis window. Each key is an ``int``:
            the number of days between enrollment and the start of the
            analysis window. Each value is a pandas DataFrame in "the
            standard format": one row per client, some metadata columns,
            plus one column per metric and sanity-check metric.
            Columns (not necessarily in order):

                * client_id (str, optional): Not necessary for
                  "happy path" analyses.
                * branch (str): The client's branch
                * other columns of ``enrollments``.
                * [metric 1]: The client's value for the first metric in
                  ``metric_list``.
                * ...
                * [metric n]: The client's value for the nth (final)
                  metric in ``metric_list``.
                * [sanity check 1]: The client's value for the first
                  sanity check metric for the first data source that
                  supports sanity checks.
                * ...
                * [sanity check n]: The client's value for the last
                  sanity check metric for the last data source that
                  supports sanity checks.
        """
        time_limits = TimeLimits.for_ts(
            self.start_date, last_date_full_data, time_series_period,
            self.num_dates_enrollment
        )

        res = self._get_per_client_data(
            enrollments, metric_list, time_limits, keep_client_id
        ).drop('mozanalysis_analysis_window_end').toPandas()

        return {
            aw.start: res[
                res['mozanalysis_analysis_window_start'] == aw.start
            ].drop('mozanalysis_analysis_window_start', axis='columns')
            for aw in time_limits.analysis_windows
        }

    def get_time_series_data_lazy(
        self, enrollments, metric_list, last_date_full_data,
        time_series_period='weekly', keep_client_id=False
    ):
        """Like ``get_time_series_data()`` but with Spark DataFrames.

        See docs for ``get_time_series_data()``.

        Sometimes the results of ``get_time_series_data()`` do not fit
        in memory. This method returns a dict of Spark DataFrames,
        instead of a dict of pandas DataFrames. This allows for
        subsequent processing to be done in Spark, or for smaller
        portions of data to be loaded into memory.
        """
        time_limits = TimeLimits.for_ts(
            self.start_date, last_date_full_data, time_series_period,
            self.num_dates_enrollment
        )

        master_df = self._get_per_client_data(
            enrollments, metric_list, time_limits, keep_client_id
        ).drop('mozanalysis_analysis_window_end').cache()

        return {
            aw.start:
                master_df.filter(
                    master_df.mozanalysis_analysis_window_start == aw.start
                ).drop('mozanalysis_analysis_window_start')
            for aw in time_limits.analysis_windows
        }

    def _get_per_client_data(
        self, enrollments, metric_list, time_limits, keep_client_id
    ):
        """Return a Spark DataFrame with metric values per-client-and-analysis-window.

        Each row contains the metric data for a (client, analysis window)
        pair.

        End users should prefer calling the public convenience methods
        ``get_per_client_data()`` and ``get_time_series_data()``,
        because they have simplified task-specific inputs, and output
        DataFrames in the standard format (one row per client).
        """
        enrollments = self._process_enrollments(enrollments, time_limits)

        enrollments = self._add_analysis_windows_to_enrollments(
            enrollments, time_limits
        )

        data_source_dfs_and_metric_col_lists = self._process_metrics(
            enrollments, metric_list
        )

        res_per_ds = [
            self._get_results_for_one_data_source(
                enrollments,
                self._process_data_source_df(ds_df, time_limits),
                mcl,
            )
            for ds_df, mcl in data_source_dfs_and_metric_col_lists.items()
        ]

        res = reduce(lambda x, y: x.join(y, enrollments.columns), res_per_ds)

        if keep_client_id:
            return res

        return res.drop(enrollments.client_id)

    def _get_results_for_one_data_source(
        self, enrollments, data_source_df, metric_column_list
    ):
        """Return a DataFrame of aggregated per-client metrics.

        Left join ``data_source_df`` to ``enrollments`` to get per-client
        data within the analysis windows, then aggregate to compute the
        requested metrics plus some sanity checks.
        """
        join_on = self._get_join_conditions(enrollments, data_source_df)

        res = enrollments.join(
            data_source_df,
            join_on,
            'left'
        ).groupBy(
            *[enrollments[c] for c in enrollments.columns]  # Yes, really.
        ).agg(
            *metric_column_list
        )

        return res

    def _get_join_conditions(self, enrollments, data_source):
        """Return a list of join conditions.

        Returns a list of boolean ``Column``s representing join
        conditions between the ``enrollments`` ``DataFrame`` and
        ``data_source``.

        In ``_get_results_for_one_data_source``, we left join
        ``enrollments`` to ``data_source`` using these join conditions
        to produce a ``DataFrame`` containing the rows from
        ``data_source`` for enrolled clients that were submitted during
        the analysis window.
        """
        # Use F.col() to avoid a bug in spark when `enrollments` is built
        # from `data_source` (SPARK-10925)
        days_since_enrollment = (
            F.unix_timestamp(F.col('submission_date_s3'), 'yyyyMMdd')
            - F.unix_timestamp(enrollments.enrollment_date, 'yyyyMMdd')
        ) / (24 * 60 * 60)

        join_on = [
            enrollments.client_id == data_source.client_id,

            # Do a quick pass aiming to efficiently filter out lots of rows:
            enrollments.enrollment_date <= F.col('submission_date_s3'),

            # Now do a more thorough pass filtering out irrelevant data:
            days_since_enrollment.between(
                enrollments.mozanalysis_analysis_window_start,
                enrollments.mozanalysis_analysis_window_end
            )
        ]

        if 'experiments' in data_source.columns:
            # Try to filter data from day of enrollment before time of enrollment.
            # If the client enrolled and unenrolled on the same day then this
            # will also filter out that day's post unenrollment data but that's
            # probably the smallest and most innocuous evil on the menu.
            join_on.append(
                (enrollments.enrollment_date != F.col('submission_date_s3'))
                | (~F.isnull(data_source.experiments[self.experiment_slug]))
            )

        return join_on

    @staticmethod
    def _add_analysis_windows_to_enrollments(enrollments, time_limits):
        """Return ``enrollments`` cross joined on analysis windows.

        When querying time series, we need to query an extra dimension
        of data (time!): each datum/value is identified by the tuple::

            (client, analysis window, metric)

        where "analysis window" identifies a point in the time series.

        In order to get 3 dimensions of data from a DB that returns
        2-dimensional ``DataFrames``, we move from a "one row per
        client" regime to a "one row per (client, analysis window)"
        regime.

        This method converts a "one row per client" ``enrollments``
        ``DataFrame`` to a "one row per (client, period)"
        ``enrollments`` ``DataFrame``, returning the Cartesian product
        of clients and analysis windows.

        An analysis window is identified in the returned ``DataFrame``
        by the pair of columns:

            * ``mozanalysis_analysis_window_start`` (int)
            * ``mozanalysis_analysis_window_end`` (int)
        """
        analysis_windows = [
            [aw.start, aw.end] for aw in time_limits.analysis_windows
        ]

        analysis_window_df = enrollments.sql_ctx.createDataFrame(
            analysis_windows,
            'mozanalysis_analysis_window_start: int, '
            'mozanalysis_analysis_window_end: int'
        )

        return enrollments.crossJoin(analysis_window_df)

    @staticmethod
    def _get_enrollments_view_normandy(spark):
        """Return a DataFrame of all normandy enrollment events.

        Filter the ``events`` table to enrollment events and transform it
        into the standard enrollments schema.

        Args:
            spark: The spark context.
        """
        events = spark.table('events')

        return events.filter(
            events.event_category == 'normandy'
        ).filter(
            events.event_method == 'enroll'
        ).select(
            events.client_id,
            events.event_string_value.alias('experiment_slug'),
            events.event_map_values.branch.alias('branch'),
            events.submission_date_s3.alias('enrollment_date'),
        )

    @staticmethod
    def _get_enrollments_view_addon(spark):
        """Return a DataFrame of all addon study enrollment events.

        Filter the ``telemetry_shield_study_parquet`` to enrollment events
        and transform it into the standard enrollments schema.

        Args:
            spark: The spark context.
        """
        tssp = spark.table('telemetry_shield_study_parquet')

        return tssp.filter(
            tssp.payload.data.study_state == 'enter'
        ).select(
            tssp.client_id,
            tssp.payload.study_name.alias('experiment_slug'),
            tssp.payload.branch.alias('branch'),
            tssp.submission.alias('enrollment_date'),
            tssp.payload.addon_version.alias('addon_version'),
        )

    @staticmethod
    def _process_enrollments(enrollments, time_limits):
        """Return ``enrollments``, filtered to the relevant dates.

        Ignore enrollments that were received after the enrollment
        period (if one was specified), else ignore enrollments for
        whom we do not have complete data for all analysis windows.

        Name the returned ``DataFrame`` 'enrollments', for consistency.
        """
        enrollments = enrollments.filter(
            enrollments.enrollment_date <= time_limits.last_enrollment_date
        )

        return enrollments.alias('enrollments')

    def _process_metrics(self, enrollments, metric_list):
        """Return a dict of lists of Columns, representing metrics.

        Each key is the DataFrame to which the Columns belong.
        """
        spark = enrollments.sql_ctx.sparkSession

        res = {}
        for m in reversed(metric_list):
            ds_df = m.data_source.get_dataframe(spark, self)

            if ds_df not in res:
                res[ds_df] = m.data_source.get_sanity_metric_cols(self, enrollments)

            res[ds_df].insert(0, m.get_col(spark, self))

        return res

    @staticmethod
    def _process_data_source_df(data_source, time_limits):
        """Return ``data_source``, filtered to the relevant dates.

        Ignore data before the analysis window of the first enrollment,
        and after the analysis window of the last enrollment.  This
        should not affect the results - it should just speed things up.

        Name the returned ``DataFrame`` 'data_source', for consistency.
        """
        for col in ['client_id', 'submission_date_s3']:
            if col not in data_source.columns:
                raise ValueError("Column '{}' missing from 'data_source'".format(col))

        return data_source.filter(
            data_source.submission_date_s3.between(
                time_limits.first_date_data_required,
                time_limits.last_date_data_required
            )
        ).alias('data_source')


@attr.s(frozen=True, slots=True)
class TimeLimits(object):
    """Internal object containing various time limits.

    Instantiated and used by the ``Experiment`` class; end users
    should not need to interact with it.

    Do not directly instantiate: use the constructors provided.

    There are several time constraints needed to specify a valid query
    for experiment data:

        * When did enrollments start?
        * When did enrollments stop?
        * How long after enrollment does the analysis window start?
        * How long is the analysis window?

    Even if these four quantities are specified directly, it is
    important to check that they are consistent with the available
    data - i.e. that we have data for the entire analysis window for
    every enrollment.

    Furthermore, there are some extra quantities that are useful for
    writing efficient queries:

        * What is the first date for which we need data from our data
          source?
        * What is the last date for which we need data from our data
          source?

    Instances of this class store all these quantities and do validation
    to make sure that they're consistent. The "store lots of overlapping
    state and validate" strategy was chosen over "store minimal state
    and compute on the fly" because different state is supplied in
    different contexts.
    """

    first_enrollment_date = attr.ib(type=str)
    last_enrollment_date = attr.ib(type=str)

    first_date_data_required = attr.ib(type=str)
    last_date_data_required = attr.ib(type=str)

    analysis_windows = attr.ib()  # type: tuple[AnalysisWindow]

    @classmethod
    def for_single_analysis_window(
        cls,
        first_enrollment_date,
        last_date_full_data,
        analysis_start_days,
        analysis_length_dates,
        num_dates_enrollment=None,
    ):
        """Return a ``TimeLimits`` instance with the following parameters

        Args:
            first_enrollment_date (str): First date on which enrollment
                events were received; the start date of the experiment.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            analysis_start_days (int): the start of the analysis window,
                measured in 'days since the client enrolled'. We ignore data
                collected outside this analysis window.
            analysis_length_days (int): the length of the analysis window,
                measured in days.
            num_dates_enrollment (int, optional): Only include this many days
                of enrollments. If ``None`` then use the maximum number of days
                as determined by the metric's analysis window and
                ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
                factor ``7`` removes weekly seasonality, and the ``+1``
                accounts for the fact that enrollment typically starts a few
                hours before UTC midnight.
        """
        analysis_window = AnalysisWindow(
            analysis_start_days, analysis_start_days + analysis_length_dates - 1
        )

        if num_dates_enrollment is None:
            last_enrollment_date = add_days(last_date_full_data, -analysis_window.end)

        else:
            last_enrollment_date = add_days(
                first_enrollment_date, num_dates_enrollment - 1
            )

            if add_days(
                last_enrollment_date, analysis_window.end
            ) > last_date_full_data:
                raise ValueError(
                    "You said you wanted {} dates of enrollment, ".format(
                        num_dates_enrollment
                    ) + "and need data from the {}th day after enrollment. ".format(
                        analysis_window.end
                    ) + "For that, you need to wait until we have data for {}.".format(
                        last_enrollment_date
                    )
                )

        first_date_data_required = add_days(
            first_enrollment_date, analysis_window.start
        )
        last_date_data_required = add_days(last_enrollment_date, analysis_window.end)

        tl = cls(
            first_enrollment_date=first_enrollment_date,
            last_enrollment_date=last_enrollment_date,
            first_date_data_required=first_date_data_required,
            last_date_data_required=last_date_data_required,
            analysis_windows=(analysis_window,),
        )
        return tl

    @classmethod
    def for_ts(
        cls,
        first_enrollment_date,
        last_date_full_data,
        time_series_period,
        num_dates_enrollment,
    ):
        """Return a ``TimeLimits`` instance for a time series.

        Args:
            first_enrollment_date (str): First date on which enrollment
                events were received; the start date of the experiment.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            time_series_period: 'daily' or 'weekly'.
            num_dates_enrollment (int): Take this many days of client
                enrollments. This is a mandatory argument because it
                determines the number of points in the time series.
        """
        if time_series_period not in ('daily', 'weekly'):
            raise ValueError("Unsupported time series period {}".format(
                time_series_period
            ))

        analysis_window_length_dates = 1 if time_series_period == 'daily' else 7

        last_enrollment_date = add_days(
            first_enrollment_date, num_dates_enrollment - 1
        )
        max_dates_of_data = date_sub(last_date_full_data, last_enrollment_date) + 1
        num_periods = max_dates_of_data // analysis_window_length_dates

        if num_periods <= 0:
            raise ValueError("Insufficient data")

        analysis_windows = tuple([
            AnalysisWindow(
                i * analysis_window_length_dates,
                (i + 1) * analysis_window_length_dates - 1
            )
            for i in range(num_periods)
        ])

        last_date_data_required = add_days(
            last_enrollment_date, analysis_windows[-1].end
        )

        return cls(
            first_enrollment_date=first_enrollment_date,
            last_enrollment_date=last_enrollment_date,
            first_date_data_required=first_enrollment_date,
            last_date_data_required=last_date_data_required,
            analysis_windows=analysis_windows,
        )

    @first_enrollment_date.validator
    def _validate_first_enrollment_date(self, attribute, value):
        assert self.first_enrollment_date <= self.last_enrollment_date
        assert self.first_enrollment_date <= self.first_date_data_required
        assert self.first_enrollment_date <= self.last_date_data_required

    @last_enrollment_date.validator
    def _validate_last_enrollment_date(self, attribute, value):
        assert self.last_enrollment_date <= self.last_date_data_required

    @first_date_data_required.validator
    def _validate_first_date_data_required(self, attribute, value):
        assert self.first_date_data_required <= self.last_date_data_required

        min_analysis_window_start = min(aw.start for aw in self.analysis_windows)
        assert self.first_date_data_required == add_days(
            self.first_enrollment_date, min_analysis_window_start
        )

    @last_date_data_required.validator
    def _validate_last_date_data_required(self, attribute, value):
        max_analysis_window_end = max(aw.end for aw in self.analysis_windows)
        assert self.last_date_data_required == add_days(
            self.last_enrollment_date, max_analysis_window_end
        )


@attr.s(frozen=True, slots=True)
class AnalysisWindow(object):
    """Represents the range of days in which to measure a metric.

    The range is measured in "days after enrollment", and is inclusive.

    For example, ``AnalysisWindow(0, 6)`` is the first week after enrollment.

    Args:
        start (int): First day of the analysis window, in days since
            enrollment.
        end (int): Final day of the analysis window, in days since
            enrollment.
    """
    start = attr.ib(type=int)
    end = attr.ib(type=int)

    @start.validator
    def _validate_start(self, attribute, value):
        assert value >= 0

    @end.validator
    def _validate_end(self, attribute, value):
        assert value >= self.start
