# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from pyspark.sql import functions as F

from mozanalysis.utils import add_days


class Experiment(object):
    """Get DataFrames of experiment data; store experiment metadata.

    The methods here query data in a way compatible with the following
    principles, which are important for experiment analysis:

    - The population of clients in each branch must have the same
        properties, aside from the intervention itself and its
        consequences; i.e. there must be no underlying bias in the
        branch populations.
    - Every client should have an equal opportunity to convert, so that
        we are measuring the same thing for each client.

    So that our analyses follow these abstract principles, we follow
    these rules:

    - Start with a list of all clients who enrolled.
    - We can filter this list of clients only based on information known
        to us at or before the time that they enrolled, because later
        information might be causally connected to the intervention.
    - For any given metric, every client gets a non-null value; we don't
        implicitly ignore anyone, even if they churned and stopped
        sending data.
    - Typically if an enrolled user no longer qualifies for enrollment,
        we'll still want to include their data in the analysis, unless
        we're explicitly using stats methods that handle censored data.


    Example usage:
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
            conv_window_start_days=0,
            conv_window_length_days=7
        )

        # Pull data into a pandas df, ready for running stats
        pres = res.toPandas()

    Args:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '20190101'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many dates
            of enrollments. If `None` then use the maximum number of dates
            as determined by the metric's conversion window and
            `last_date_full_data`. Typically `7n+1`, e.g. `8`. The
            factor '7' removes weekly seasonality, and the `+1` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        addon_version (str): The version of the experiment addon. Some
            addon experiment slugs get reused - in those cases we need to
            filter on the addon version also.

    Attributes:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '20190101'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many days
            of enrollments. If `None` then use the maximum number of days
            as determined by the metric's conversion window and
            `last_date_full_data`. Typically `7n+1`, e.g. `8`. The
            factor '7' removes weekly seasonality, and the `+1` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        addon_version (str): The version of the experiment addon. Some
            addon experiment slugs get reused - in those cases we need to
            filter on the addon version also.
    """
    def __init__(
        self, experiment_slug, start_date, num_dates_enrollment=None,
        addon_version=None
    ):
        # Let's be conservative about storing state - it doesn't belong
        # in this class. Treat these attributes as immutable.
        # These attributes are stored because it would be a PITA not to
        # store them:
        #   - they are required by both `get_enrollments()` and
        #       `get_per_client_data()`
        #   - if you were accidentally inconsistent with their values
        #       then you would hit trouble quickly!
        self.experiment_slug = experiment_slug
        self.start_date = start_date
        self.num_dates_enrollment = num_dates_enrollment
        self.addon_version = addon_version

    def get_enrollments(self, spark, study_type='pref_flip', end_date=None):
        """Return a DataFrame of enrolled clients.

        This works for pref-flip and addon studies.

        The underlying queries are different for pref-flip vs addon
        studies, because as of 2019/04/02, branch information isn't
        reliably available in the `events` table for addon experiments:
        branch may be NULL for all enrollments. The enrollment
        information for them is most reliably available in
        `telemetry_shield_study_parquet`. Once this issue is resolved,
        we will probably start using normandy events for all desktop
        studies.
        Ref: https://bugzilla.mozilla.org/show_bug.cgi?id=1536644

        Args:
            spark: The spark context.
            study_type (str): One of the following strings:
                - 'pref_flip'
                - 'addon'
            end_date (str, optional): Ignore enrollments after this
                date: for faster queries on stale experiments. If you
                set `num_dates_enrollment` then do not set this; at best
                it would be redundant, at worst it's contradictory.

        Returns:
            A Spark DataFrame of enrollment data. One row per
            enrollment. Columns:
                - client_id (str)
                - enrollment_date (str): e.g. '20190329'
                - branch (str)
        """
        if study_type == 'pref_flip':
            enrollments = self._get_enrollments_view_normandy(spark)

        elif study_type == 'addon':
            enrollments = self._get_enrollments_view_addon(spark, self.addon_version)

        # elif study_type == 'glean':
        #     raise NotImplementedError

        else:
            raise ValueError("Unrecognized study_type: {}".format(study_type))

        enrollments = enrollments.filter(
            enrollments.enrollment_date >= self.start_date
        ).filter(
            enrollments.experiment_slug == self.experiment_slug
        )

        if self.num_dates_enrollment is not None:
            assert end_date is None, "Possibly contradictory instructions!"
            enrollments = enrollments.filter(
                enrollments.enrollment_date <= self._get_scheduled_max_enrollment_date()
            )
        elif end_date is not None:
            enrollments = enrollments.filter(
                enrollments.enrollment_date <= end_date
            )

        enrollments.cache()

        return enrollments

    def get_per_client_data(
        self, enrollments, df, metric_list, last_date_full_data, conv_window_start_days,
        conv_window_length_days, keep_client_id=False
    ):
        """Return a DataFrame containing per-client metric values.

        Args:
            enrollments: A spark DataFrame of enrollments, like the one
                returned by `self.get_enrollments()`.
            df: A spark DataFrame containing the data needed to calculate
                the metrics. Could be `main_summary` or `clients_daily`.
                _Don't_ use `experiments`; as of 2019/04/02 it drops data
                collected after people self-unenroll, so unenrolling users
                will appear to churn. Must have at least the following
                columns:
                    - client_id (str)
                    - submission_date_s3 (str)
                    - data columns referred to in `metric_list`
                Ideally also has:
                    - experiments (map)
            metric_list: A list of columns that aggregate and compute
                metrics over data grouped by `(client_id, branch)`, e.g.
                `[F.coalesce(F.sum(df.metric_name), F.lit(0)).alias('metric_name')]`
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            conv_window_start_days (int): the start of the conversion window,
                measured in 'days since the user enrolled'. We ignore data
                collected outside this conversion window.
            conv_window_length_days (int): the length of the conversion window,
                measured in days.
            keep_client_id (bool): Whether to return a `client_id` column.
                Defaults to False to reduce memory usage of the results.

        Returns:
            A spark DataFrame of experiment data. One row per `client_id`.
            One or two metadata columns, then one column per metric in
            `metric_list`. Then one column per sanity-check metric.
            Columns:
                - client_id (str, optional): Not necessary for
                    "happy path" analyses.
                - branch (str): The client's branch
                - [metric 1]: The client's value for the first metric in
                    `metric_list`.
                - ...
                - [metric n]: The client's value for the nth (final)
                    metric in `metric_list`.
                - [sanity check 1]: The client's value for the first
                    sanity check metric.
                - ...
                - [sanity check n]: The client's value for the last
                    sanity check metric.

            This format - the schema plus there being one row per
            enrolled client, regardless of whether the client has data
            in `df` - was agreed upon by the DS team, and is the
            standard format for queried experimental data.
        """
        req_dates_of_data = conv_window_start_days + conv_window_length_days

        if self.num_dates_enrollment is None:
            self._print_enrollment_window(last_date_full_data, req_dates_of_data)

        # TODO accuracy: can/should we switch from submission_date_s3 to when the
        # events actually happened?

        enrollments = self.filter_enrollments_for_conv_window(
            enrollments, last_date_full_data, req_dates_of_data
        )

        df = self.filter_df_for_conv_window(
            df, last_date_full_data, conv_window_start_days, conv_window_length_days
        )

        sanity_metrics = self._get_telemetry_sanity_check_metrics(enrollments, df)

        res = enrollments.join(
            df,
            [
                # TODO perf: would it be faster if we enforce a join on sample_id?
                enrollments.client_id == df.client_id,

                # TODO accuracy: once we can rely on
                #   `df.experiments[self.experiment_slug]`
                # existing even after unenrollment, we could start joining on
                # branch to reduce problems associated with split client_ids:
                # enrollments.branch == df.experiments[self.experiment_slug]

                # Do a quick pass aiming to efficiently filter out lots of rows:
                enrollments.enrollment_date <= df.submission_date_s3,

                # Now do a more thorough pass filtering out irrelevant data:
                # TODO perf: what is a more efficient way to do this?
                (
                    (
                        F.unix_timestamp(df.submission_date_s3, 'yyyyMMdd')
                        - F.unix_timestamp(enrollments.enrollment_date, 'yyyyMMdd')
                    ) / (24 * 60 * 60)
                ).between(
                    conv_window_start_days,
                    conv_window_start_days + conv_window_length_days - 1
                ),

                # Try to filter data from day of enrollment before time of enrollment.
                # If the client enrolled and unenrolled on the same day then this
                # will also filter out that day's post unenrollment data but that's
                # probably the smallest and most innocuous evil on the menu.
                (
                    (enrollments.enrollment_date != df.submission_date_s3)
                    | (~F.isnull(df.experiments[self.experiment_slug]))
                ),

            ],
            'left'
        ).groupBy(
            enrollments.client_id, enrollments.branch
        ).agg(
            *(metric_list + sanity_metrics)
        )
        if keep_client_id:
            return res
        else:
            return res.drop(enrollments.client_id)

    @staticmethod
    def _get_enrollments_view_normandy(spark):
        """Return a DataFrame of all normandy enrollment events.

        Filter the `events` table to enrollment events and transform it
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
    def _get_enrollments_view_addon(spark, addon_version=None):
        """Return a DataFrame of all addon study enrollment events.

        Filter the `telemetry_shield_study_parquet` to enrollment events
        and transform it into the standard enrollments schema.

        Args:
            spark: The spark context.
        """
        tssp = spark.table('telemetry_shield_study_parquet')

        if addon_version is not None:
            tssp = tssp.filter(
                tssp.payload.addon_version == addon_version
            )

        return tssp.filter(
            tssp.payload.data.study_state == 'enter'
        ).select(
            tssp.client_id,
            tssp.payload.study_name.alias('experiment_slug'),
            tssp.payload.branch.alias('branch'),
            tssp.submission.alias('enrollment_date'),
        )

    def _print_enrollment_window(self, last_date_full_data, req_dates_of_data):
        """Print the enrollment dates being used.

        We need `req_dates_of_data` days of post-enrollment data per user.
        This places limits on how early we can run certain analyses.
        This method calculates and prints these limits.

        Args:
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            req_dates_of_data (int): The minimum number of dates of
                post-enrollment data required to have data for the user
                for the entire conversion window.
        """
        print("Taking enrollments between {} and {}".format(
            self.start_date,
            self._get_last_enrollment_date(
                last_date_full_data, req_dates_of_data
            )
        ))

    def _get_scheduled_max_enrollment_date(self):
        """Return the last enrollment date, according to the plan."""
        assert self.num_dates_enrollment is not None

        return add_days(self.start_date, self.num_dates_enrollment - 1)

    def _get_last_enrollment_date(self, last_date_full_data, req_dates_of_data):
        """Return the date of the final used enrollment.

        We need `req_dates_of_data` days of post-enrollment data per user.
        This and `last_date_full_data` put constraints on the enrollment
        period. This method checks these constraints are feasible, and
        compatible with any manually supplied enrollment period.

        Args:
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '20190322'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            req_dates_of_data (int): The minimum number of dates of
                post-enrollment data required to have data for the user
                for the entire conversion window.
        """
        last_enrollment_with_data = add_days(
            last_date_full_data, -(req_dates_of_data - 1)
        )

        if self.num_dates_enrollment is None:
            assert last_enrollment_with_data >= self.start_date, \
                "No users have had time to convert yet"

            return last_enrollment_with_data

        else:
            intended_last_enrollment = self._get_scheduled_max_enrollment_date()
            assert last_enrollment_with_data >= intended_last_enrollment, \
                "You said you wanted {} dates of enrollment, ".format(
                    self.num_dates_enrollment
                ) + "but your conversion window of {} days won't have ".format(
                    req_dates_of_data
                ) + "complete data until we have the data for {}.".format(
                    add_days(intended_last_enrollment, req_dates_of_data - 1)
                )
            return intended_last_enrollment

    def _get_last_data_date(self, last_date_full_data, req_dates_of_data):
        """Return the date of the final used datum."""
        if self.num_dates_enrollment is None:
            assert last_date_full_data >= add_days(
                self.start_date, req_dates_of_data - 1
            ), "No users have had time to convert yet"

            return last_date_full_data

        last_required_data_date = add_days(
            self.start_date,
            self.num_dates_enrollment - 1 + req_dates_of_data - 1
        )

        # If I did the math right, this should be equivalent to the
        # check in _get_last_enrollment_date()
        assert last_required_data_date <= last_date_full_data, \
            "You said you wanted {} dates of enrollment, ".format(
                self.num_dates_enrollment
            ) + "but your conversion time of {} days won't have ".format(
                req_dates_of_data
            ) + "complete data until we have the data for {}.".format(
                last_required_data_date
            )

        return last_required_data_date

    def filter_enrollments_for_conv_window(
        self, enrollments, last_date_full_data, req_dates_of_data
    ):
        """Return the enrollments, filtered to the relevant dates."""
        return enrollments.filter(
            # Ignore clients that might convert in the future
            enrollments.enrollment_date <= self._get_last_enrollment_date(
                last_date_full_data, req_dates_of_data
            )
        )

    def filter_df_for_conv_window(
        self, df, last_date_full_data, conv_window_start_days, conv_window_length_days
    ):
        """Return the df, filtered to the relevant dates.

        This should not affect the results - it should just speed things
        up.
        """
        return df.filter(
            # Ignore data before the conversion window of the first enrollment
            df.submission_date_s3 >= add_days(
                self.start_date, conv_window_start_days
            )
        ).filter(
            # Ignore data after the conversion window of the last enrollment,
            # and data after the specified `last_date_full_data`
            df.submission_date_s3 <= self._get_last_data_date(
                last_date_full_data, conv_window_start_days + conv_window_length_days
            )
        )

    def _get_telemetry_sanity_check_metrics(self, enrollments, df):
        """Return aggregations that check for problems with a client."""

        # TODO: Once we know what form the metrics library will take,
        # we should move the below metric definitions and documentation
        # into it.

        if dict(df.dtypes).get('experiments') != 'map<string,string>':
            # Not all tables have an experiments map - can't make these checks.
            return []

        return [

            # Check to see whether the client_id is also enrolled in other branches
            # E.g. indicates cloned profiles. Fraction of such users should be
            # small, and similar between branches.
            F.max(F.coalesce((
                df.experiments[self.experiment_slug] != enrollments.branch
            ).astype('int'), F.lit(0))).alias('has_contradictory_branch'),

            # Check to see whether the client_id was sending data in the conversion
            # window that wasn't tagged as being part of the experiment. Indicates
            # either a client_id clash, or the client unenrolling. Fraction of such
            # users should be small, and similar between branches.
            F.max(F.coalesce((
                ~F.isnull(df.experiments)
                & F.isnull(df.experiments[self.experiment_slug])
            ).astype('int'), F.lit(0))).alias('has_non_enrolled_data'),
        ]
