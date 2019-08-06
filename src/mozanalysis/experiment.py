# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr
from pyspark.sql import Column, functions as F

from mozanalysis.utils import add_days


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

        cd = spark.table('clients_daily')

        experiment = Experiment(
            experiment_slug='pref-flip-defaultoncookierestrictions-1506704',
            start_date='20181215',
            num_dates_enrollment=8
        )
        enrollments = experiment.get_enrollments(spark)

        res = experiment.get_per_client_data(
            enrollments,
            cd,
            [
                F.sum(F.coalesce(
                    cd.active_hours_sum, F.lit(0)
                )).alias('active_hours'),
                F.sum(F.coalesce(
                    cd.scalar_parent_browser_engagement_total_uri_count_sum,
                    F.lit(0)
                )).alias('uri_count'),
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
        self, enrollments, data_source, metric_list, last_date_full_data,
        analysis_start_days, analysis_length_days, keep_client_id=False
    ):
        """Return a DataFrame containing per-client metric values.

        Args:
            enrollments: A spark DataFrame of enrollments, like the one
                returned by ``self.get_enrollments()``.
            data_source: A spark DataFrame containing the data needed to
                calculate the metrics. Could be ``main_summary`` or
                ``clients_daily``. *Don't* use ``experiments``; as of 2019/04/02
                it drops data collected after people self-unenroll, so
                unenrolling users will appear to churn. Must have at least
                the following columns:

                * client_id (str)
                * submission_date_s3 (str)
                * data columns referred to in ``metric_list``

                Ideally also has:

                * experiments (map): At present this is used to exclude
                  pre-enrollment ping data collected on enrollment
                  day. Once it or its successor reliably tags data
                  from all enrolled users, even post-unenroll, we'll
                  also join on it to exclude data from duplicate
                  ``client_id``\\s that are not enrolled in the same
                  branch.

            metric_list: A list of columns that aggregate and compute
                metrics over data grouped by ``(client_id, branch)``, e.g.::

                    [F.coalesce(F.sum(
                        data_source.metric_name
                    ), F.lit(0)).alias('metric_name')]

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
            One or two metadata columns, then one column per metric in
            ``metric_list``. Then one column per sanity-check metric.
            Columns:

                * client_id (str, optional): Not necessary for
                  "happy path" analyses.
                * branch (str): The client's branch
                * [metric 1]: The client's value for the first metric in
                  ``metric_list``.
                * ...
                * [metric n]: The client's value for the nth (final)
                  metric in ``metric_list``.
                * [sanity check 1]: The client's value for the first
                  sanity check metric.
                * ...
                * [sanity check n]: The client's value for the last
                  sanity check metric.

            This format - the schema plus there being one row per
            enrolled client, regardless of whether the client has data
            in ``data_source`` - was agreed upon by the DS team, and is the
            standard format for queried experimental data.
        """
        time_limits = TimeLimits.create(
            self.start_date, last_date_full_data, analysis_start_days,
            analysis_length_days, self.num_dates_enrollment
        )

        enrollments = self._process_enrollments(enrollments, time_limits)

        data_source = self._process_data_source(data_source, time_limits)

        join_on = self._get_join_conditions(enrollments, data_source, time_limits)

        sanity_metrics = self._get_telemetry_sanity_check_metrics(
            enrollments, data_source
        )

        res = enrollments.join(
            data_source,
            join_on,
            'left'
        ).groupBy(
            enrollments.client_id, enrollments.branch
        ).agg(
            *(metric_list + sanity_metrics)
        )
        if keep_client_id:
            return res

        return res.drop(enrollments.client_id)

    def _get_join_conditions(self, enrollments, data_source, time_limits):
        """Return a list of join conditions.

        In ``_get_results_for_one_data_source``, we join ``enrollments``
        to ``data_source``. This method returns the list of join
        conditions.
        """
        join_on = [
            # TODO perf: would it be faster if we enforce a join on sample_id?
            enrollments.client_id == data_source.client_id,

            # TODO accuracy: once we can rely on
            #   `data_source.experiments[self.experiment_slug]`
            # existing even after unenrollment, we could start joining on
            # branch to reduce problems associated with split client_ids:
            # enrollments.branch == data_source.experiments[self.experiment_slug]

            # Do a quick pass aiming to efficiently filter out lots of rows:
            # Use F.col() to avoid a bug in spark when `enrollments` is built
            # from `data_source` (SPARK-10925)
            enrollments.enrollment_date <= F.col('submission_date_s3'),

            # Now do a more thorough pass filtering out irrelevant data:
            # TODO perf: what is a more efficient way to do this?
            (
                (
                    F.unix_timestamp(F.col('submission_date_s3'), 'yyyyMMdd')
                    - F.unix_timestamp(enrollments.enrollment_date, 'yyyyMMdd')
                ) / (24 * 60 * 60)
            ).between(
                time_limits.analysis_window_start,
                time_limits.analysis_window_end
            ),
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
        whom we do not have data for the entire analysis window.
        """
        return enrollments.filter(
            enrollments.enrollment_date <= time_limits.last_enrollment_date
        ).alias('enrollments')

    @staticmethod
    def _process_data_source(data_source, time_limits):
        """Return ``data_source``, filtered to the relevant dates.

        Ignore data before the analysis window of the first enrollment,
        and after the analysis window of the last enrollment.  This
        should not affect the results - it should just speed things up.
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

    def _get_telemetry_sanity_check_metrics(self, enrollments, data_source):
        """Return aggregations that check for problems with a client."""

        # TODO: Once we know what form the metrics library will take,
        # we should move the below metric definitions and documentation
        # into it.

        if dict(data_source.dtypes).get('experiments') != 'map<string,string>':
            # Not all tables have an experiments map - can't make these checks.
            return []

        return [

            # Check to see whether the client_id is also enrolled in other branches
            # E.g. indicates cloned profiles. Fraction of such users should be
            # small, and similar between branches.
            F.max(F.coalesce((
                data_source.experiments[self.experiment_slug] != enrollments.branch
            ).astype('int'), F.lit(0))).alias('has_contradictory_branch'),

            # Check to see whether the client_id was sending data in the analysis
            # window that wasn't tagged as being part of the experiment. Indicates
            # either a client_id clash, or the client unenrolling. Fraction of such
            # users should be small, and similar between branches.
            F.max(F.coalesce((
                ~F.isnull(data_source.experiments)
                & F.isnull(data_source.experiments[self.experiment_slug])
            ).astype('int'), F.lit(0))).alias('has_non_enrolled_data'),
        ]


@attr.s(frozen=True)
class TimeLimits(object):
    """Internal object containing various time limits.

    Instantiated and used by the ``Experiment`` class; end users
    should not need to interact with it.

    Do not directly instantiate: use ``TimeLimits.create()``.

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

    analysis_window_start = attr.ib()
    """The integer number of days between enrollment and the start of
    the analysis window, represented either as an ``int`` or as a
    Column of integers in the ``enrollments`` DataFrame."""

    analysis_window_end = attr.ib()
    """The integer number of days between enrollment and the end of
    the analysis window, represented either as an ``int`` or as a
    Column of integers in the ``enrollments`` DataFrame."""

    analysis_window_length_dates = attr.ib(type=int)
    """The number of dates in the analysis window"""

    first_date_data_required = attr.ib(type=str)
    last_date_data_required = attr.ib(type=str)

    @classmethod
    def create(
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
                factor '7' removes weekly seasonality, and the ``+1`` accounts
                for the fact that enrollment typically starts a few hours
                before UTC midnight.
        """
        analysis_end_days = analysis_start_days + analysis_length_dates - 1

        if num_dates_enrollment is None:
            last_enrollment_date = add_days(last_date_full_data, -analysis_end_days)

        else:
            last_enrollment_date = add_days(
                first_enrollment_date, num_dates_enrollment - 1
            )

            if add_days(last_enrollment_date, analysis_end_days) > last_date_full_data:
                raise ValueError(
                    "You said you wanted {} dates of enrollment, ".format(
                        num_dates_enrollment
                    ) + "and need data from the {}th day after enrollment. ".format(
                        analysis_end_days
                    ) + "For that, you need to wait until we have data for {}.".format(
                        last_enrollment_date
                    )
                )

        first_date_data_required = add_days(first_enrollment_date, analysis_start_days)
        last_date_data_required = add_days(last_enrollment_date, analysis_end_days)

        tl = cls(
            first_enrollment_date=first_enrollment_date,
            last_enrollment_date=last_enrollment_date,
            analysis_window_start=analysis_start_days,
            analysis_window_end=analysis_end_days,
            analysis_window_length_dates=analysis_length_dates,
            first_date_data_required=first_date_data_required,
            last_date_data_required=last_date_data_required
        )
        return tl

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

    @analysis_window_start.validator
    def _validate_analysis_window_start(self, attribute, value):
        if not isinstance(self.analysis_window_start, Column):
            assert not isinstance(self.analysis_window_end, Column)
            assert self.analysis_window_start >= 0
            assert self.first_date_data_required == add_days(
                self.first_enrollment_date, self.analysis_window_start
            )

    @analysis_window_length_dates.validator
    def _validate_analysis_window_length_dates(self, attribute, value):
        assert self.analysis_window_length_dates >= 1

    @analysis_window_end.validator
    def _validate_analysis_window_end(self, attribute, value):
        if not isinstance(self.analysis_window_end, Column):
            assert self.analysis_window_end == \
                self.analysis_window_start + self.analysis_window_length_dates - 1
            assert self.last_date_data_required == add_days(
                self.last_enrollment_date, self.analysis_window_end
            )
