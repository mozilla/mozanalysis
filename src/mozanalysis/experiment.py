# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, cast

import attr
from metric_config_parser import AnalysisUnit
from typing_extensions import assert_never

from mozanalysis import APPS
from mozanalysis.bq import BigQueryContext, sanitize_table_name_for_bq
from mozanalysis.config import ConfigLoader
from mozanalysis.metrics import AnalysisBasis, DataSource, Metric
from mozanalysis.segments import Segment, SegmentDataSource
from mozanalysis.types import IncompatibleAnalysisUnit
from mozanalysis.utils import add_days, date_sub, hash_ish

if TYPE_CHECKING:
    from pandas import DataFrame

    from mozanalysis.exposure import ExposureSignal

logger = logging.getLogger(__name__)


class EnrollmentsQueryType(str, Enum):
    CIRRUS = "cirrus"
    FENIX_FALLBACK = "fenix-fallback"
    NORMANDY = "normandy"
    GLEAN_EVENT = "glean-event"


@attr.s(frozen=True, slots=True)
class Experiment:
    """Query experiment data; store experiment metadata.

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


    Example usage (in a colab notebook)::

        from google.colab import auth
        auth.authenticate_user()
        print('Authenticated')

        from mozanalysis.experiment import Experiment
        from mozanalysis.bq import BigQueryContext
        from mozanalysis.config import ConfigLoader

        active_hours = ConfigLoader.get_metric("active_hours", "firefox_desktop")
        uri_count = ConfigLoader.get_metric("uri_count", "firefox_desktop")

        bq_context = BigQueryContext(
            dataset_id='your-dataset-id',  # e.g. mine's flawrence
            project_id='moz-fx-data-bq-data-science'  # this is the default anyway
        )

        experiment = Experiment(
            experiment_slug='pref-fingerprinting-protections-retention-study-release-70',
            start_date='2019-10-29',
            num_dates_enrollment=8
        )

        # Run the query and get the results as a DataFrame
        res = experiment.get_single_window_data(
            bq_context,
            [
                active_hours,
                uri_count
            ],
            last_date_full_data='2019-12-01',
            analysis_start_days=0,
            analysis_length_days=7
        )

    Args:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '2019-01-01'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many dates
            of enrollments. If ``None`` then use the maximum number of dates
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        app_id (str, optional): For a Glean app, the name of the BigQuery
            dataset derived from its app ID, like `org_mozilla_firefox`.
        app_name (str, optional): The Glean app name, like `fenix`.
        analysis_unit (AnalysisUnit, optional):  the "unit" of analysis,
            which defines an experimental unit. For example: `CLIENT`
            for mobile experiments or `GROUP` for desktop experiments.  Is used
            as the join key when building queries and sub-unit level data is
            aggregated up to that level. Defaults to `AnalysisUnit.CLIENT`
            unless specified

    Attributes:
        experiment_slug (str): Name of the study, used to identify
            the enrollment events specific to this study.
        start_date (str): e.g. '2019-01-01'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many days
            of enrollments. If ``None`` then use the maximum number of days
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
    """

    experiment_slug = attr.ib(type=str, validator=attr.validators.instance_of(str))
    start_date = attr.ib()
    num_dates_enrollment = attr.ib(default=None)
    app_id = attr.ib(default=None)
    app_name = attr.ib(default=None)
    analysis_unit = attr.ib(
        type=AnalysisUnit,
        default=AnalysisUnit.CLIENT,
        validator=attr.validators.instance_of(AnalysisUnit),
    )

    def get_app_name(self):
        """
        Determine the correct app name.

        If no explicit app name has been passed into Experiment, lookup app name from
        a pre-defined list. (this is deprecated)
        """
        if self.app_name is None:
            logger.warning(
                "Experiment without `app_name` is deprecated. "
                + "Please specify an app_name explicitly"
            )
            app_name = next(key for key, value in APPS.items() if self.app_id in value)
            if app_name is None:
                raise Exception(f"No app name for app_id {self.app_id}")
        return self.app_name

    def get_single_window_data(
        self,
        bq_context: BigQueryContext,
        metric_list: list,
        last_date_full_data: str,
        analysis_start_days: int,
        analysis_length_days: int,
        enrollments_query_type: EnrollmentsQueryType = EnrollmentsQueryType.NORMANDY,
        custom_enrollments_query: str | None = None,
        custom_exposure_query: str | None = None,
        exposure_signal: ExposureSignal | None = None,
        segment_list=None,
    ) -> DataFrame:
        """Return a DataFrame containing per-client metric values.

        Also store them in a permanent table in BigQuery. The name of
        this table will be printed. Subsequent calls to this function
        will simply read the results from this table.

        Args:
            bq_context (BigQueryContext): BigQuery configuration and client.
            metric_list (list of mozanalysis.metric.Metric or str): The metrics
                to analyze.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '2019-03-22'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            analysis_start_days (int): the start of the analysis window,
                measured in 'days since the client enrolled'. We ignore data
                collected outside this analysis window.
            analysis_length_days (int): the length of the analysis window,
                measured in days.
            enrollments_query_type (EnrollmentsQueryType):
                ('normandy', 'glean-event', 'cirrus', or 'fenix-fallback')
                Specifies the query type to use to get the experiment's
                enrollments, unless overridden by
                ``custom_enrollments_query``.
            custom_enrollments_query (str): A full SQL query that
                will generate the `enrollments` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_enrolled_events`.

                WARNING: this query's results must be uniquely keyed by
                (client_id, branch), or else your results will be subtly
                wrong.

            custom_exposure_query (str):  A full SQL query that
                will generate the `exposures` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_exposure_events`.

                If not provided, the exposure will be determined based on
                `exposure_signal`, if provided, or Normandy and Nimbus exposure events.
                `custom_exposure_query` takes precedence over `exposure_signal`.

            exposure_signal (ExposureSignal): Optional signal definition of when a
                client has been exposed to the experiment. If not provided,
                the exposure will be determined based on Normandy exposure events
                for desktop and Nimbus exposure events for Fenix and iOS.
            segment_list (list of mozanalysis.segment.Segment or str): The user
                segments to study.

        Returns:
            A pandas DataFrame of experiment data. One row per ``client_id``.
            Some metadata columns, then one column per metric in
            ``metric_list``, and one column per sanity-check metric.
            Columns (not necessarily in order):

                * client_id (str): Not necessary for "happy path" analyses.
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
            self.start_date,
            last_date_full_data,
            analysis_start_days,
            analysis_length_days,
            self.num_dates_enrollment,
        )

        enrollments_sql = self.build_enrollments_query(
            time_limits=time_limits,
            enrollments_query_type=enrollments_query_type,
            custom_enrollments_query=custom_enrollments_query,
            custom_exposure_query=custom_exposure_query,
            exposure_signal=exposure_signal,
            segment_list=segment_list,
        )

        enrollments_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    last_date_full_data,
                    "enrollments",
                    self.experiment_slug,
                    hash_ish(enrollments_sql),
                ]
            )
        )

        bq_context.run_query(enrollments_sql, enrollments_table_name)

        metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            enrollments_table=bq_context.fully_qualify_table_name(
                enrollments_table_name
            ),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join([last_date_full_data, self.experiment_slug, hash_ish(metrics_sql)])
        )

        return bq_context.run_query(metrics_sql, full_res_table_name).to_dataframe()

    def get_time_series_data(
        self,
        bq_context: BigQueryContext,
        metric_list: list,
        last_date_full_data: str,
        time_series_period: str = "weekly",
        enrollments_query_type: EnrollmentsQueryType = EnrollmentsQueryType.NORMANDY,
        custom_enrollments_query: str | None = None,
        custom_exposure_query: str | None = None,
        exposure_signal: ExposureSignal | None = None,
        segment_list=None,
    ) -> TimeSeriesResult:
        """Return a TimeSeriesResult with per-client metric values.

        Roughly equivalent to looping over :meth:`.get_single_window_data`
        with different analysis windows, and reorganising the results.

        Args:
            bq_context (BigQueryContext): BigQuery configuration and client.
            metric_list (list of mozanalysis.metric.Metric):
                The metrics to analyze.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '2019-03-22'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            time_series_period ('daily' or 'weekly'): How long each
                analysis window should be.
            enrollments_query_type (EnrollmentsQueryType):
                ('normandy', 'glean-event', 'cirrus', or 'fenix-fallback')
                Specifies the query type to use to get the experiment's
                enrollments, unless overridden by
                ``custom_enrollments_query``.
            custom_enrollments_query (str): A full SQL query that
                will generate the `enrollments` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_enrolled_events`.

                WARNING: this query's results must be uniquely keyed by
                (client_id, branch), or else your results will be subtly
                wrong.

            custom_exposure_query (str): A full SQL query that
                will generate the `exposures` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_exposure_events`.

                If not provided, the exposure will be determined based on
                `exposure_signal`, if provided, or Normandy and Nimbus exposure events.
                `custom_exposure_query` takes precedence over `exposure_signal`.

            exposure_signal (ExposureSignal): Optional signal definition of when a
                client has been exposed to the experiment. If not provided,
                the exposure will be determined based on Normandy exposure events
                for desktop and Nimbus exposure events for Fenix and iOS.
            segment_list (list of mozanalysis.segment.Segment): The user
                segments to study.

        Returns:
            A :class:`mozanalysis.experiment.TimeSeriesResult` object,
            which may be used to obtain a
            pandas DataFrame of per-client metric data, for each
            analysis window. Each DataFrame is a pandas DataFrame in
            "the standard format": one row per client, some metadata
            columns, plus one column per metric and sanity-check metric.
            Its columns (not necessarily in order):

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
            self.start_date,
            last_date_full_data,
            time_series_period,
            self.num_dates_enrollment,
        )

        enrollments_sql = self.build_enrollments_query(
            time_limits=time_limits,
            enrollments_query_type=enrollments_query_type,
            custom_enrollments_query=custom_enrollments_query,
            custom_exposure_query=custom_exposure_query,
            exposure_signal=exposure_signal,
            segment_list=segment_list,
        )

        enrollments_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    last_date_full_data,
                    "enrollments",
                    self.experiment_slug,
                    hash_ish(enrollments_sql),
                ]
            )
        )

        bq_context.run_query(enrollments_sql, enrollments_table_name)

        metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            enrollments_table=bq_context.fully_qualify_table_name(
                enrollments_table_name
            ),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join([last_date_full_data, self.experiment_slug, hash_ish(metrics_sql)])
        )

        bq_context.run_query(metrics_sql, full_res_table_name)

        return TimeSeriesResult(
            fully_qualified_table_name=bq_context.fully_qualify_table_name(
                full_res_table_name
            ),
            analysis_windows=time_limits.analysis_windows,
        )

    def build_enrollments_query(
        self,
        time_limits: TimeLimits,
        enrollments_query_type: EnrollmentsQueryType = EnrollmentsQueryType.NORMANDY,
        custom_enrollments_query: str | None = None,
        custom_exposure_query: str | None = None,
        exposure_signal: ExposureSignal | None = None,
        segment_list=None,
        sample_size: int = 100,
    ) -> str:
        """Return a SQL query for querying enrollment and exposure data.

        Args:
            time_limits (TimeLimits): An object describing the
                interval(s) to query
            enrollments_query_type (EnrollmentsQueryType):
                ('normandy', 'glean-event', 'cirrus', or 'fenix-fallback')
                Specifies the query type to use to get the experiment's
                enrollments, unless overridden by
                ``custom_enrollments_query``.
            custom_enrollments_query (str): A full SQL query that
                will generate the `enrollments` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_enrolled_events`.

                WARNING: this query's results must be uniquely keyed by
                (client_id, branch), or else your results will be subtly
                wrong.
            custom_exposure_query (str): A full SQL query that
                will generate the `exposures` common table expression
                used in the main query. The query must produce the columns
                `client_id`, `branch`, `enrollment_date`, and `num_exposure_events`.

            exposure_signal (ExposureSignal): Optional signal definition of when a
                client has been exposed to the experiment

            segment_list (list of mozanalysis.segment.Segment or str): The user
                segments to study.

            sample_size (int): Optional integer percentage of clients, used for
                downsampling enrollments. Default 100.

        Returns:
            A string containing a BigQuery SQL expression.
        """
        sample_size = sample_size or 100

        enrollments_query = custom_enrollments_query or self._build_enrollments_query(
            time_limits,
            enrollments_query_type,
            sample_size,
        )

        if exposure_signal:
            exposure_query = custom_exposure_query or exposure_signal.build_query(
                time_limits, self.analysis_unit
            )
        else:
            exposure_query = custom_exposure_query or self._build_exposure_query(
                time_limits,
                enrollments_query_type,
            )

        segments_query = self._build_segments_query(
            segment_list,
            time_limits,
        )

        return f"""
            WITH raw_enrollments AS ({enrollments_query}),
            segmented_enrollments AS ({segments_query}),
            exposures AS ({exposure_query})

            SELECT
                se.*,
                e.* EXCEPT ({self.analysis_unit.value}, branch)
            FROM segmented_enrollments se
            LEFT JOIN exposures e
            USING ({self.analysis_unit.value}, branch)
        """

    def build_metrics_query(
        self,
        metric_list: list,
        time_limits: TimeLimits,
        enrollments_table: str,
        analysis_basis=AnalysisBasis.ENROLLMENTS,
        exposure_signal: ExposureSignal | None = None,
    ) -> str:
        """Return a SQL query for querying metric data.

        For interactive use, prefer :meth:`.get_time_series_data` or
        :meth:`.get_single_window_data`, according to your use case,
        which will run the query for you and return a materialized
        dataframe.

        The optional ``exposure_signal`` parameter allows to check if
        clients have received the exposure signal during enrollment or
        after. When using the exposures analysis basis, metrics will
        be computed for these clients.

        Args:
            metric_list (list of mozanalysis.metric.Metric or str):
                The metrics to analyze.
            time_limits (TimeLimits): An object describing the
                interval(s) to query
            enrollments_table (str): The name of the enrollments table
            basis (AnalysisBasis): Use exposures as basis for calculating
                metrics if True, otherwise use enrollments.
            exposure_signal (Optional[ExposureSignal]): Optional exposure
                signal parameter that will be used for computing metrics
                for certain analysis bases (such as exposures).

        Returns:
            A string containing a BigQuery SQL expression.

        Building this query is the main goal of this module.
        """
        analysis_windows_query = self._build_analysis_windows_query(
            time_limits.analysis_windows
        )

        metrics_columns, metrics_joins = self._build_metrics_query_bits(
            metric_list, time_limits, analysis_basis, exposure_signal
        )

        if exposure_signal and analysis_basis != AnalysisBasis.ENROLLMENTS:
            exposure_query = f"""
            SELECT * FROM (
                {exposure_signal.build_query(time_limits, self.analysis_unit)}
            )
            WHERE num_exposure_events > 0
            """
        else:
            exposure_query = """
                SELECT
                    *
                FROM raw_enrollments e
            """

        return """
        WITH analysis_windows AS (
            {analysis_windows_query}
        ),
        raw_enrollments AS (
            -- needed by "exposures" sub query
            SELECT
                e.*,
                aw.*
            FROM `{enrollments_table}` e
            CROSS JOIN analysis_windows aw
        ),
        exposures AS ({exposure_query}),
        enrollments AS (
            SELECT
                e.* EXCEPT (exposure_date, num_exposure_events),
                x.exposure_date,
                x.num_exposure_events
            FROM exposures x
                RIGHT JOIN raw_enrollments e
                USING ({id_column}, branch)
        )
        SELECT
            enrollments.*,
            {metrics_columns}
        FROM enrollments
        {metrics_joins}
        """.format(
            analysis_windows_query=analysis_windows_query,
            exposure_query=exposure_query,
            metrics_columns=",\n        ".join(metrics_columns),
            metrics_joins="\n".join(metrics_joins),
            enrollments_table=enrollments_table,
            id_column=self.analysis_unit.value,
        )

    @staticmethod
    def _build_analysis_windows_query(analysis_windows) -> str:
        """Return SQL to construct a table of analysis windows.

        To query a time series, we construct a table of analysis windows
        and cross join it with the enrollments table to get one row per
        pair of client and analysis window.

        This method writes the SQL to define the analysis window table.
        """
        return "\n        UNION ALL\n        ".join(
            f"(SELECT {aw.start} AS analysis_window_start, {aw.end} AS analysis_window_end)"  # noqa:E501
            for aw in analysis_windows
        )

    def _build_enrollments_query(
        self,
        time_limits: TimeLimits,
        enrollments_query_type: EnrollmentsQueryType,
        sample_size: int = 100,
    ) -> str:
        """Return SQL to query a list of enrollments and their branches"""
        if enrollments_query_type == EnrollmentsQueryType.NORMANDY:
            return self._build_enrollments_query_normandy(
                time_limits,
                sample_size,
            )
        elif enrollments_query_type == EnrollmentsQueryType.GLEAN_EVENT:
            if not self.app_id:
                raise ValueError(
                    "App ID must be defined for building Glean enrollments query"
                )
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Glean enrollments currently only support client_id analysis units"
                )
            return self._build_enrollments_query_glean_event(
                time_limits, self.app_id, sample_size
            )
        elif enrollments_query_type == EnrollmentsQueryType.FENIX_FALLBACK:
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Fenix fallback enrollments currently only support client_id analysis units"  # noqa: E501
                )
            return self._build_enrollments_query_fenix_baseline(
                time_limits, sample_size
            )
        elif enrollments_query_type == EnrollmentsQueryType.CIRRUS:
            if not self.app_id:
                raise ValueError(
                    "App ID must be defined for building Cirrus enrollments query"
                )
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Cirrus enrollments currently only support client_id analysis units"
                )
            return self._build_enrollments_query_cirrus(time_limits, self.app_id)
        else:
            assert_never(enrollments_query_type)

    def _build_exposure_query(
        self,
        time_limits: TimeLimits,
        exposure_query_type: EnrollmentsQueryType,
    ) -> str:
        """Return SQL to query a list of exposures and their branches"""
        if exposure_query_type == EnrollmentsQueryType.NORMANDY:
            return self._build_exposure_query_normandy(time_limits)
        elif exposure_query_type == EnrollmentsQueryType.GLEAN_EVENT:
            if not self.app_id:
                raise ValueError(
                    "App ID must be defined for building Glean exposures query"
                )
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Glean exposures currently only support client_id analysis units"
                )
            return self._build_exposure_query_glean_event(time_limits, self.app_id)
        elif exposure_query_type == EnrollmentsQueryType.FENIX_FALLBACK:
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Fenix fallback exposures currently only support client_id analysis units"  # noqa: E501
                )
            return self._build_exposure_query_glean_event(
                time_limits, "org_mozilla_firefox"
            )
        elif exposure_query_type == EnrollmentsQueryType.CIRRUS:
            if not self.app_id:
                raise ValueError(
                    "App ID must be defined for building Cirrus exposures query"
                )
            if not self.analysis_unit == AnalysisUnit.CLIENT:
                raise IncompatibleAnalysisUnit(
                    "Cirrus exposures currently only support client_id analysis units"
                )
            return self._build_exposure_query_glean_event(
                time_limits,
                self.app_id,
                client_id_field='mozfun.map.get_key(event.extra, "user_id")',
                event_category="cirrus_events",
            )
        else:
            assert_never(exposure_query_type)

    def _build_enrollments_query_normandy(
        self,
        time_limits: TimeLimits,
        sample_size: int = 100,
    ) -> str:
        """Return SQL to query enrollments for a normandy experiment"""
        return f"""
        SELECT
            e.{self.analysis_unit.value},
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
                BETWEEN '{time_limits.first_enrollment_date}' AND '{time_limits.last_enrollment_date}'
            AND e.event_string_value = '{self.experiment_slug}'
            AND e.sample_id < {sample_size}
        GROUP BY e.{self.analysis_unit.value}, branch
            """  # noqa:E501

    def _build_enrollments_query_fenix_baseline(
        self, time_limits: TimeLimits, sample_size: int = 100
    ) -> str:
        """Return SQL to query enrollments for a Fenix no-event experiment
        If enrollment events are available for this experiment, then you
        can take a better approach than this method. But in the absence
        of enrollment events (e.g. in a Mako-based experiment, which
        does not send enrollment events), you need to fall back to using
        ``ping_info.experiments`` to get a list of who is in what branch
        and when they enrolled.
        """
        # Try to ignore users who enrolled early - but only consider a
        # 7 day window

        return """
        SELECT
            b.client_info.client_id AS client_id,
            mozfun.map.get_key(
                b.ping_info.experiments,
                '{experiment_slug}'
            ).branch,
            DATE(MIN(b.submission_timestamp)) AS enrollment_date,
            COUNT(b.submission_date) AS num_enrollment_events
        FROM `moz-fx-data-shared-prod.{dataset}.baseline` b
        WHERE
            b.client_info.client_id IS NOT NULL AND
            DATE(b.submission_timestamp)
                BETWEEN DATE_SUB('{first_enrollment_date}', INTERVAL 7 DAY)
                AND '{last_enrollment_date}'
            AND mozfun.map.get_key(
                b.ping_info.experiments,
                '{experiment_slug}'
            ).branch IS NOT NULL
            AND b.sample_id < {sample_size}
        GROUP BY client_id, branch
        HAVING enrollment_date >= '{first_enrollment_date}'
            """.format(
            experiment_slug=self.experiment_slug,
            first_enrollment_date=time_limits.first_enrollment_date,
            last_enrollment_date=time_limits.last_enrollment_date,
            dataset=self.app_id or "org_mozilla_firefox",
            sample_size=sample_size,
        )

    def _build_enrollments_query_glean_event(
        self, time_limits: TimeLimits, dataset: str, sample_size: int = 100
    ) -> str:
        """Return SQL to query enrollments for a Glean no-event experiment

        If enrollment events are available for this experiment, then you
        can take a better approach than this method. But in the absence
        of enrollment events (e.g. in a Mako-based experiment, which
        does not send enrollment events), you need to fall back to using
        ``ping_info.experiments`` to get a list of who is in what branch
        and when they enrolled.
        """

        return f"""
            SELECT events.client_info.client_id AS client_id,
                mozfun.map.get_key(
                    e.extra,
                    'branch'
                ) AS branch,
                DATE(MIN(events.submission_timestamp)) AS enrollment_date,
                COUNT(events.submission_timestamp) AS num_enrollment_events
            FROM `moz-fx-data-shared-prod.{self.app_id or dataset}.events` events,
            UNNEST(events.events) AS e
            WHERE
                events.client_info.client_id IS NOT NULL AND
                DATE(events.submission_timestamp)
                BETWEEN '{time_limits.first_enrollment_date}' AND '{time_limits.last_enrollment_date}'
                AND e.category = "nimbus_events"
                AND mozfun.map.get_key(e.extra, "experiment") = '{self.experiment_slug}'
                AND e.name = 'enrollment'
                AND sample_id < {sample_size}
            GROUP BY client_id, branch
            """  # noqa:E501

    def _build_enrollments_query_cirrus(
        self, time_limits: TimeLimits, dataset: str
    ) -> str:
        """Return SQL to query enrollments for a Cirrus experiment (uses Glean)

        If enrollment events are available for this experiment, then you
        can take a better approach than this method. But in the absence
        of enrollment events (e.g. in a Mako-based experiment, which
        does not send enrollment events), you need to fall back to using
        ``ping_info.experiments`` to get a list of who is in what branch
        and when they enrolled.
        """

        return f"""
            SELECT
                mozfun.map.get_key(e.extra, "user_id") AS client_id,
                mozfun.map.get_key(
                    e.extra,
                    'branch'
                ) AS branch,
                DATE(MIN(events.submission_timestamp)) AS enrollment_date,
                COUNT(events.submission_timestamp) AS num_enrollment_events
            FROM `moz-fx-data-shared-prod.{self.app_id or dataset}.enrollment` events,
            UNNEST(events.events) AS e
            WHERE
                mozfun.map.get_key(e.extra, "user_id") IS NOT NULL AND
                DATE(events.submission_timestamp)
                BETWEEN '{time_limits.first_enrollment_date}' AND '{time_limits.last_enrollment_date}'
                AND e.category = "cirrus_events"
                AND mozfun.map.get_key(e.extra, "experiment") = '{self.experiment_slug}'
                AND e.name = 'enrollment'
                AND client_info.app_channel = 'production'
            GROUP BY client_id, branch
            """  # noqa:E501

    def _build_exposure_query_normandy(self, time_limits: TimeLimits) -> str:
        """Return SQL to query exposures for a normandy experiment"""
        return f"""
        SELECT
            e.{self.analysis_unit.value},
            e.branch,
            min(e.submission_date) AS exposure_date,
            COUNT(e.submission_date) AS num_exposure_events
        FROM raw_enrollments re
        LEFT JOIN (
            SELECT
                {self.analysis_unit.value},
                `mozfun.map.get_key`(event_map_values, 'branchSlug') AS branch,
                submission_date
            FROM
                `moz-fx-data-shared-prod.telemetry.events`
            WHERE
                event_category = 'normandy'
                AND (event_method = 'exposure' OR event_method = 'expose')
                AND submission_date
                    BETWEEN '{time_limits.first_enrollment_date}' AND '{time_limits.last_enrollment_date}'
                AND event_string_value = '{self.experiment_slug}'
        ) e
        ON re.{self.analysis_unit.value} = e.{self.analysis_unit.value} AND
            re.branch = e.branch AND
            e.submission_date >= re.enrollment_date
        GROUP BY e.{self.analysis_unit.value}, e.branch
            """  # noqa: E501

    def _build_exposure_query_glean_event(
        self,
        time_limits: TimeLimits,
        dataset: str,
        client_id_field: str = "client_info.client_id",
        event_category: str = "nimbus_events",
    ) -> str:
        """Return SQL to query exposures for a Glean no-event experiment"""
        return f"""
            SELECT
                exposures.client_id,
                exposures.branch,
                DATE(MIN(exposures.submission_date)) AS exposure_date,
                COUNT(exposures.submission_date) AS num_exposure_events
            FROM raw_enrollments re
            LEFT JOIN (
                SELECT
                    {client_id_field} AS client_id,
                    mozfun.map.get_key(event.extra, 'branch') AS branch,
                    DATE(events.submission_timestamp) AS submission_date
                FROM
                    `moz-fx-data-shared-prod.{self.app_id or dataset}.events` events,
                    UNNEST(events.events) AS event
                WHERE
                    DATE(events.submission_timestamp)
                    BETWEEN '{time_limits.first_enrollment_date}' AND '{time_limits.last_enrollment_date}'
                    AND event.category = '{event_category}'
                    AND mozfun.map.get_key(
                        event.extra,
                        "experiment") = '{self.experiment_slug}'
                    AND (event.name = 'expose' OR event.name = 'exposure')
            ) exposures
            ON re.client_id = exposures.client_id AND
                re.branch = exposures.branch AND
                exposures.submission_date >= re.enrollment_date
            GROUP BY client_id, branch
            """  # noqa: E501

    def _build_metrics_query_bits(
        self,
        metric_list: list[Metric | str],
        time_limits: TimeLimits,
        analysis_basis=AnalysisBasis.ENROLLMENTS,
        exposure_signal: ExposureSignal | None = None,
    ) -> tuple[list[str], list[str]]:
        """Return lists of SQL fragments corresponding to metrics."""
        metrics: list[Metric] = []
        for metric in metric_list:
            if isinstance(metric, str):
                metrics.append(ConfigLoader.get_metric(metric, self.get_app_name()))
            else:
                metrics.append(metric)

        ds_metrics = self._partition_metrics_by_data_source(metrics)
        ds_metrics = cast(dict[DataSource, list[Metric]], ds_metrics)
        ds_metrics = {
            ds: metrics + ds.get_sanity_metrics(self.experiment_slug)
            for ds, metrics in ds_metrics.items()
        }

        metrics_columns = []
        metrics_joins = []

        for i, ds in enumerate(ds_metrics.keys()):
            query_for_metrics = ds.build_query(
                ds_metrics[ds],
                time_limits,
                self.experiment_slug,
                self.app_id,
                analysis_basis,
                self.analysis_unit,
                exposure_signal,
            )

            metrics_joins.append(
                f"""    LEFT JOIN (
            {query_for_metrics}
            ) ds_{i} USING ({self.analysis_unit.value}, branch, analysis_window_start, analysis_window_end)
                    """  # noqa: E501
            )

            for m in ds_metrics[ds]:
                metrics_columns.append(f"ds_{i}.{m.name}")

        return metrics_columns, metrics_joins

    def _partition_segments_by_data_source(
        self, segment_list: list[Segment]
    ) -> dict[SegmentDataSource, list[Segment]]:
        """Return a dict mapping segment data sources to segment lists."""
        data_sources = {s.data_source for s in segment_list}

        return {
            ds: [s for s in segment_list if s.data_source == ds] for ds in data_sources
        }

    def _partition_metrics_by_data_source(
        self, metric_list: list[Metric]
    ) -> dict[DataSource, list[Metric]]:
        """Return a dict mapping data sources to metric/segment lists."""
        data_sources = {m.data_source for m in metric_list}

        return {
            ds: [m for m in metric_list if m.data_source == ds] for ds in data_sources
        }

    def _build_segments_query(
        self,
        segment_list: list[Segment],
        time_limits: TimeLimits,
    ) -> str:
        """Build a query adding segment columns to the enrollments view.

        The query takes a ``raw_enrollments`` view, and defines a new
        view by adding one non-NULL boolean column per segment. It does
        not otherwise tamper with the ``raw_enrollments`` view.
        """

        # Do similar things to what we do for metrics, but in a less
        # ostentatious place, since people are likely to come to the
        # source code asking how metrics work, but less likely to
        # arrive with "how segments work" as their first question.

        segments_columns, segments_joins = self._build_segments_query_bits(
            cast(list[Segment | str], segment_list) or [], time_limits
        )

        return """
        SELECT
            raw_enrollments.*,
            {segments_columns}
        FROM raw_enrollments
        {segments_joins}
        """.format(
            segments_columns=",\n        ".join(segments_columns),
            segments_joins="\n".join(segments_joins),
        )

    def _build_segments_query_bits(
        self,
        segment_list: list[Segment | str],
        time_limits: TimeLimits,
    ) -> tuple[list[str], list[str]]:
        """Return lists of SQL fragments corresponding to segments."""

        # resolve segment slugs
        segments: list[Segment] = []
        for segment in segment_list:
            if isinstance(segment, str):
                segments.append(ConfigLoader.get_segment(segment, self.get_app_name()))
            else:
                segments.append(segment)

        ds_segments = self._partition_segments_by_data_source(segments)

        segments_columns = []
        segments_joins = []

        for i, ds in enumerate(ds_segments.keys()):
            query_for_segments = ds.build_query(
                ds_segments[ds],
                time_limits,
                self.experiment_slug,
                self.app_id,
                self.analysis_unit,
            )
            segments_joins.append(
                f"""    LEFT JOIN (
        {query_for_segments}
        ) ds_{i} USING ({self.analysis_unit.value}, branch)
                """
            )

            for m in ds_segments[ds]:
                segments_columns.append(f"ds_{i}.{m.name}")

        return segments_columns, segments_joins


@attr.s(frozen=True, slots=True)
class TimeLimits:
    """Expresses time limits for different kinds of analysis windows.

    Instantiated and used by the :class:`Experiment` class; end users
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

    analysis_windows = attr.ib()  # type: tuple[AnalysisWindow,...]

    @classmethod
    def for_single_analysis_window(
        cls,
        first_enrollment_date: str,
        last_date_full_data: str,
        analysis_start_days: int,
        analysis_length_dates: int,
        num_dates_enrollment: int | None = None,
    ) -> TimeLimits:
        """Return a ``TimeLimits`` instance with the following parameters

        Args:
            first_enrollment_date (str): First date on which enrollment
                events were received; the start date of the experiment.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '2019-03-22'. If you want to ignore
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

        first_date_data_required = add_days(
            first_enrollment_date, analysis_window.start
        )
        last_date_data_required = add_days(last_enrollment_date, analysis_window.end)

        if last_date_data_required > last_date_full_data:
            raise ValueError(
                f"You said you wanted {num_dates_enrollment} dates of enrollment, "
                + f"and need data from the {analysis_window.end}th day after enrollment. "  # noqa: E501
                + f"For that, you need to wait until we have data for {last_date_data_required}."  # noqa:E501
            )

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
        first_enrollment_date: str,
        last_date_full_data: str,
        time_series_period: str,
        num_dates_enrollment: int,
    ) -> TimeLimits:
        """Return a ``TimeLimits`` instance for a time series.

        Args:
            first_enrollment_date (str): First date on which enrollment
                events were received; the start date of the experiment.
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '2019-03-22'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            time_series_period: 'daily' or 'weekly'.
            num_dates_enrollment (int): Take this many days of client
                enrollments. This is a mandatory argument because it
                determines the number of points in the time series.
        """
        period_duration = {"daily": 1, "weekly": 7, "28_day": 28}

        if time_series_period not in period_duration:
            raise ValueError(f"Unsupported time series period {time_series_period}")

        if num_dates_enrollment <= 0:
            raise ValueError("Number of enrollment dates must be a positive number")

        analysis_window_length_dates = period_duration[time_series_period]

        last_enrollment_date = add_days(first_enrollment_date, num_dates_enrollment - 1)
        max_dates_of_data = date_sub(last_date_full_data, last_enrollment_date) + 1
        num_periods = max_dates_of_data // analysis_window_length_dates

        if num_periods <= 0:
            raise ValueError("Insufficient data")

        analysis_windows = tuple(
            [
                AnalysisWindow(
                    i * analysis_window_length_dates,
                    (i + 1) * analysis_window_length_dates - 1,
                )
                for i in range(num_periods)
            ]
        )

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
        assert self.first_enrollment_date <= self.last_enrollment_date, (
            f"first enrollment date of {self.first_enrollment_date} ",
            f"was not on or before last enrollment date of {self.last_enrollment_date}",
        )

    @first_date_data_required.validator
    def _validate_first_date_data_required(self, attribute, value):
        assert self.first_date_data_required <= self.last_date_data_required, (
            f"first date data required of {self.first_date_data_required} was not on ",
            f"or before last date data required of {self.last_date_data_required}",
        )

        min_analysis_window_start = min(aw.start for aw in self.analysis_windows)
        observation_period_start = add_days(
            self.first_enrollment_date, min_analysis_window_start
        )
        assert self.first_date_data_required == observation_period_start, (
            f"first date data required of {self.first_date_data_required} ",
            f"did not match computed start of observation {observation_period_start}",
        )

    @last_date_data_required.validator
    def _validate_last_date_data_required(self, attribute, value):
        max_analysis_window_end = max(aw.end for aw in self.analysis_windows)
        observation_period_end = add_days(
            self.last_enrollment_date, max_analysis_window_end
        )
        assert self.last_date_data_required == observation_period_end, (
            f"last date data required of {self.last_date_data_required} ",
            f"did not match computed end of observation {observation_period_end}",
        )


@attr.s(frozen=True, slots=True)
class AnalysisWindow:
    """Represents the range of days in which to measure a metric.

    The range is measured in "days relative enrollment", and is inclusive.

    For example, ``AnalysisWindow(0, 6)`` is the first week after enrollment
    and `AnalysisWindow(-8,-1)` is the week before enrollment

    Args:
        start (int): First day of the analysis window, in days relative
            to enrollment start. 0 indicates the date of enrollment.
            Positive numbers are after enrollment, negative are before.
            Must be the same sign as `end` (zero counts as positive)
        end (int): Final day of the analysis window, in days relative
            to enrollment start. 0 indicates the date of enrollment.
            Positive numbers are after enrollment, negative are before.
            Must be the same sign as `start` (zero counts as positive).
    """

    start = attr.ib(type=int)
    end = attr.ib(type=int)

    @start.validator
    def _validate_start(self, attribute, value):
        assert (value >= 0 and self.end >= 0) or (value < 0 and self.end < 0)

    @end.validator
    def _validate_end(self, attribute, value):
        assert value >= self.start
        assert (value >= 0 and self.start >= 0) or (value < 0 and self.start < 0)


@attr.s(frozen=True, slots=True)
class TimeSeriesResult:
    """Result from a time series query.

    For each analysis window, this object lets us get a dataframe in
    "the standard format" (one row per client).

    Example usage::

        result_dict = dict(time_series_result.items(bq_context))
        window_0 = result_dict[0]

    ``window_0`` would then be a pandas DataFrame of results for the
    analysis window starting at day 0. ``result_dict`` would be a
    dictionary of all such DataFrames, keyed by the start days of
    their analysis windows.

    Or, to load only one analysis window into RAM::

        window_0 = time_series_result.get(bq_context, 0)
    """

    fully_qualified_table_name = attr.ib(type=str)
    analysis_windows = attr.ib(type=tuple[AnalysisWindow, ...])
    analysis_unit = attr.ib(type=AnalysisUnit, default=AnalysisUnit.CLIENT)

    def get(self, bq_context: BigQueryContext, analysis_window) -> DataFrame:
        """Get the DataFrame for a specific analysis window.

        N.B. this makes a BigQuery query each time it is run; caching
        results is your responsibility.

        Args:
            bq_context (BigQueryContext)
            analysis_window (AnalysisWindow or int): The analysis
                window, or its start day as an int.
        """
        if isinstance(analysis_window, int):
            try:
                analysis_window = next(
                    aw for aw in self.analysis_windows if aw.start == analysis_window
                )
            except StopIteration as err:
                raise KeyError(
                    f"AnalysisWindow not found with start of {analysis_window}"
                ) from err

        return bq_context.run_query(
            self._build_analysis_window_subset_query(analysis_window)
        ).to_dataframe()

    def get_full_data(self, bq_context: BigQueryContext) -> DataFrame:
        """Get the full DataFrame from TimeSeriesResult.

        This DataFrame has a row for each client for each period of the time
        series and may be very large. A warning will print the size of data
        to be downloaded.

        Args:
            bq_context (BigQueryContext)
        """
        size = self._get_table_size(bq_context)

        print(f"Downloading {self.fully_qualified_table_name} ({size} GB)")

        table = bq_context.client.get_table(self.fully_qualified_table_name)
        return bq_context.client.list_rows(table).to_dataframe()

    def get_aggregated_data(
        self,
        bq_context: BigQueryContext,
        metric_list: list,
        aggregate_function: str = "AVG",
    ) -> tuple[DataFrame, int]:
        """Results from a time series query, aggregated over analysis windows
        by a SQL aggregate function.

        This DataFrame has a row for each analysis window, with a column
        for each metric in the supplied metric_list.

        Args:
            bq_context (BigQueryContext)
            metric_list (list of mozanalysis.metrics.Metric)
            aggregate_fuction (str)
        """

        return (
            bq_context.run_query(
                self._build_aggregated_data_query(metric_list, aggregate_function)
            ).to_dataframe(),
            bq_context.run_query(self._table_sample_size_query())
            .to_dataframe()["population_size"]
            .values[0],
        )

    def keys(self):
        return [aw.start for aw in self.analysis_windows]

    def items(self, bq_context):
        for aw in self.analysis_windows:
            yield (aw.start, self.get(bq_context, aw))

    def _get_table_size(self, bq_context: BigQueryContext) -> float:
        """
        Get table size in memory for table being requested by `get_full_data`.
        """

        table_info = self.fully_qualified_table_name.split(".")

        query = f"""
                SELECT
                    SUM(size_bytes)/pow(10,9) AS size
                FROM
                    `{table_info[0]}.{table_info[1]}`.__TABLES__
                WHERE
                  table_id = '{table_info[2]}'
                """

        size = bq_context.run_query(query).to_dataframe()

        return size["size"].iloc[0].round(2)

    def _build_analysis_window_subset_query(
        self, analysis_window: AnalysisWindow
    ) -> str:
        """Return SQL for partitioning time series results.

        When we query data for a time series, we query it for all
        points of the time series, and we store this in a table.

        This method returns SQL to query this table to obtain results
        in "the standard format" for a single analysis window.
        """
        except_clause = (
            f"{self.analysis_unit.value}, analysis_window_start, analysis_window_end"
        )
        return f"""
            SELECT * EXCEPT ({except_clause})
            FROM {self.fully_qualified_table_name}
            WHERE analysis_window_start = {analysis_window.start}
            AND analysis_window_end = {analysis_window.end}
        """

    def _build_aggregated_data_query(
        self, metric_list: list[Metric], aggregate_function: str
    ) -> str:
        return """
        SELECT
            analysis_window_start,
            analysis_window_end,
            {agg_metrics}
        FROM
            {full_table_name}
        GROUP BY
            analysis_window_start, analysis_window_end
        ORDER BY
            analysis_window_start
        """.format(
            agg_metrics=",\n            ".join(
                f"{aggregate_function}({m.name}) AS {m.name}" for m in metric_list
            ),
            full_table_name=self.fully_qualified_table_name,
        )

    def _table_sample_size_query(
        self, client_id_column: str = AnalysisUnit.CLIENT.value
    ) -> str:
        return f"""
        SELECT
            COUNT(*) as population_size
        FROM
            (SELECT DISTINCT
                {client_id_column}
            FROM
                {self.fully_qualified_table_name})
        """

    @analysis_windows.validator
    def _check_analysis_windows(self, attribute, value):
        if len(value) != len({aw.start for aw in value}):
            raise ValueError("Each analysis window must start on a different day")
