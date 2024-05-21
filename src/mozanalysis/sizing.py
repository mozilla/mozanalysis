# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from datetime import date
from typing import TYPE_CHECKING

import attr

from mozanalysis.bq import BigQueryContext, sanitize_table_name_for_bq
from mozanalysis.experiment import (
    AnalysisWindow,
    TimeLimits,
    TimeSeriesResult,
    add_days,
)
from mozanalysis.utils import hash_ish

if TYPE_CHECKING:
    from pandas import DataFrame

    from mozanalysis.metrics import DataSource, Metric
    from mozanalysis.segments import Segment, SegmentDataSource


@attr.s(frozen=False, slots=True)
class HistoricalTarget:
    """Query historical data.

    The methods here query data in a way compatible with the following
    principles, which are important for experiment analysis:

    Example usage (in a colab notebook)::

        from google.colab import auth
        auth.authenticate_user()
        print('Authenticated')

        from mozanalysis.config import ConfigLoader

        active_hours = ConfigLoader.get_metric("active_hours", "firefox_desktop")
        uri_count = ConfigLoader.get_metric("uri_count", "firefox_desktop")

        new_or_resurrected_v3 = ConfigLoader.get_segment("new_or_resurrected_v3",
                                                            "firefox_desktop")
        allweek_regular_v1 = ConfigLoader.get_segment("allweek_regular_v1",
                                                            "firefox_desktop")

        bq_context = BigQueryContext(
            dataset_id='mbowerman',  # e.g. mine's mbowerman
            project_id='moz-fx-data-bq-data-science'  # this is the default anyway
        )

        ht = HistoricalTarget(
            experiment_name='my_test_name',
            start_date='2021-01-01',
            num_dates_enrollment=2,
            analysis_length=4
        )

        # Run the query and get the results as a DataFrame
        res = ht.get_single_window_data(
            bq_context,
            metric_list = [
                active_hours,
                uri_count
            ],
            target_list = [
                allweek_regular_v1,
                new_or_resurrected_v3
            ]
        )

        targets_sql = ht.get_targets_sql()

        metrics_sql = ht.get_metrics_sql()

    Args:
        experiment_name (str): A simple name of this analysis, which is used
            to create table names when saving results in BigQuery.
        start_date (str): e.g. '2019-01-01'. First date for historical data to be
            retrieved.
        analysis_length (int): Number of days to include for analysis
        num_dates_enrollment (int, default 7): Number of days to consider to
            enroll clients in the analysis.
        continuous_enrollment (bool): Indicates if the analysis dates
            and enrollment should be overlap; clients that satisfy the
            target conditions at any point in the analysis will be
            included for the entire window when calculating metrics.
        app_id (str, optional): For a Glean app, the name of the BigQuery
            dataset derived from its app ID, like `org_mozilla_firefox`.

    Attributes:
        experiment_name (str): Name of the study, used in naming tables
            related to this analysis.
        start_date (str): e.g. '2019-01-01'. First date on which enrollment
            events were received.
        num_dates_enrollment (int): Number of days to consider to enroll clients
            in the analysis.
        analysis_length (int, optional): Number of days to include for analysis
    """

    experiment_name = attr.ib()
    start_date = attr.ib()
    analysis_length = attr.ib()
    num_dates_enrollment = attr.ib(default=7)
    continuous_enrollment = attr.ib(default=False)
    app_id = attr.ib(default=None)

    _metrics_sql = attr.ib(default=None)
    _targets_sql = attr.ib(default=None)

    def get_single_window_data(
        self,
        bq_context: BigQueryContext,
        metric_list: list[Metric],
        target_list: list[Segment] | None = None,
        custom_targets_query: str | None = None,
        replace_tables: bool = False,
    ) -> DataFrame:
        """Return a DataFrame containing per-client metric values.

        Also store them in a permanent table in BigQuery. The name of
        this table will be printed. Subsequent calls to this function
        will simply read the results from this table.

        Args:
            bq_context (BigQueryContext): BigQuery configuration and client.
            metric_list (list of mozanalysis.metric.Metric): The metrics
                to analyze.
            target_list (list of mozanalysis.segments.Segment): The targets
                that define clients to be included in the analysis, based on
                inclusion in a defined user segment. For each Segment included
                in the list, client IDs are identified in the
                SegmentDataSource that based on the select_for statement in
                the Segment. Client IDs that satisfy the select_for
                statement in every Segment in the target_list will be
                returned by the query.
            custom_targets_query (str): A full SQL query to be used
                in the main query; must include a client_id column
                to join to metrics tables and an enrollment_date:

                N.B. this query's results must be uniquely keyed by
                (client_id, enrollment_date), or else your results will be
                subtly wrong.
            replace_tables (bool): If True, delete tables that exist in
                BigQuery with the same name; used to rerun analyses under
                the same experiment name with different settings

        Returns:
            A pandas DataFrame of experiment data. One row per ``client_id``.
            Some metadata columns, then one column per metric in
            ``metric_list``, and one column per sanity-check metric.
            Columns (not necessarily in order):

                * client_id (str): Not necessary for "happy path" analyses.
                * Analysis window (int): Start and end of analysis window, in
                  days of entire analysis included in the window
                * [metric 1]: The client's value for the first metric in
                  ``metric_list``.
                * ...
                * [metric n]: The client's value for the nth (final)
                  metric in ``metric_list``.
        """
        if not (custom_targets_query or target_list):
            raise ValueError(
                "Either custom_target_query or target_list must be provided"
            )

        # validate metric_list and target_list are from the same source
        # filter out el.app_name is None, there are cases where metric-hub
        # was not used and it's up to users to ensure the sources match
        metric_list_sources = {el.app_name for el in metric_list if el.app_name}
        target_list_sources = {el.app_name for el in target_list if el.app_name}

        if len(metric_list_sources) > 1:
            warnings.warn("metric_list contains multiple metric-hub apps", stacklevel=1)

        if len(target_list_sources) > 1:
            warnings.warn("target_list contains multiple metric-hub apps", stacklevel=1)

        if (
            metric_list_sources
            and target_list_sources
            and metric_list_sources != target_list_sources
        ):
            warnings.warn(
                "metric_list and target_list metric-hub apps do not match",
                stacklevel=1,
            )

        last_date_full_data = add_days(
            self.start_date, self.num_dates_enrollment + self.analysis_length - 1
        )

        today = date.today().strftime("%Y-%m-%d")
        if last_date_full_data >= today:
            raise ValueError(
                f"Based on the start date, {self.start_date}"
                + f", with {self.num_dates_enrollment} days of enrollment "
                + f"and analysis of length {self.analysis_length} days, "
                + f"the last day of analysis is {last_date_full_data}"
                + ", which is in the future."
            )

        if self.continuous_enrollment:
            time_limits = ContinuousEnrollmentTimeLimits.for_single_analysis_window(
                self.start_date,
                self.analysis_length,
            )

        else:
            time_limits = TimeLimits.for_single_analysis_window(
                self.start_date,
                last_date_full_data,
                1,
                self.analysis_length,
                self.num_dates_enrollment,
            )

        self._targets_sql = self.build_targets_query(
            time_limits=time_limits,
            target_list=target_list,
            custom_targets_query=custom_targets_query,
        )

        targets_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    "targets",
                    self.experiment_name,
                    hash_ish(self._targets_sql),
                ]
            )
        )

        bq_context.run_query(self._targets_sql, targets_table_name, replace_tables)

        self._metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            targets_table=bq_context.fully_qualify_table_name(targets_table_name),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    self.experiment_name,
                    hash_ish(self._metrics_sql),
                ]
            )
        )

        output = bq_context.run_query(
            self._metrics_sql, full_res_table_name, replace_tables
        ).to_dataframe()

        for metric_obj in metric_list:
            if all(output[metric_obj.name] == 0):
                warnings.warn(
                    (
                        f"Metric {metric_obj.name} is all 0, which may indicate"
                        + " segments and metric do not have a common app"
                    ),
                    stacklevel=1,
                )
        return output

    def get_time_series_data(
        self,
        bq_context: BigQueryContext,
        metric_list: list[Metric],
        time_series_period: str = "weekly",
        custom_targets_query: str | None = None,
        target_list: list[HistoricalTarget] | None = None,
        replace_tables: bool = False,
    ) -> TimeSeriesResult:
        """Return a TimeSeriesResult with per-client metric values.

        Roughly equivalent to looping over :meth:`.get_single_window_data`
        with different analysis windows, and reorganising the results.

        Args:
            bq_context (BigQueryContext): BigQuery configuration and client.
            metric_list (list of mozanalysis.metric.Metric): The metrics
                to analyze.
            time_series_period (str): Period of the time series for which to
                retrieve data. Options are daily, weekly, and 28_day.
            target_list (list of mozanalysis.segments.Segment): The targets
                that define clients to be included in the analysis, based on
                inclusion in a defined user segment. For each Segment included
                in the list, client IDs are identified in the
                SegmentDataSource that based on the select_for statement in
                the Segment. Client IDs that satisfy the select_for
                statement in every Segment in the target_list will be
                returned by the query.
            custom_targets_query (str): A full SQL query to be used
                in the main query; must include a client_id column
                to join to metrics tables and an enrollment_date:

                N.B. this query's results must be uniquely keyed by
                (client_id, enrollment_date), or else your results will be
                subtly wrong.
            replace_tables (bool): If True, delete tables that exist in
                BigQuery with the same name; used to rerun analyses under
                the same experiment name with different settings

        Returns:
            A :class:`mozanalysis.experiment.TimeSeriesResult` object,
            which may be used to obtain a
            pandas DataFrame of per-client metric data, for each
            analysis window. Each DataFrame is a pandas DataFrame in
            "the standard format": one row per client, some metadata
            columns, plus one column per metric and sanity-check metric.
            Its columns (not necessarily in order):

                * enrollment date and client_id
                * [metric 1]: The client's value for the first metric in
                  ``metric_list``.
                * ...
                * [metric n]: The client's value for the nth (final)
                  metric in ``metric_list``.
        """
        last_date_full_data = add_days(
            self.start_date, self.num_dates_enrollment + self.analysis_length - 1
        )

        today = date.today().strftime("%Y-%m-%d")
        if last_date_full_data >= today:
            raise ValueError(
                f"Based on the start date, {self.start_date}"
                + f", with {self.num_dates_enrollment} days of enrollment "
                + f"and analysis of length {self.analysis_length} days, "
                + f"the last day of analysis is {last_date_full_data}"
                + ", which is in the future."
            )

        time_limits = TimeLimits.for_ts(
            self.start_date,
            last_date_full_data,
            time_series_period,
            self.num_dates_enrollment,
        )

        self._targets_sql = self.build_targets_query(
            time_limits=time_limits,
            target_list=target_list,
            custom_targets_query=custom_targets_query,
        )

        targets_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    "targets",
                    self.experiment_name,
                    hash_ish(self._targets_sql),
                ]
            )
        )

        bq_context.run_query(self._targets_sql, targets_table_name, replace_tables)

        self._metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            targets_table=bq_context.fully_qualify_table_name(targets_table_name),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    self.experiment_name,
                    hash_ish(self._metrics_sql),
                ]
            )
        )

        bq_context.run_query(self._metrics_sql, full_res_table_name, replace_tables)

        return TimeSeriesResult(
            fully_qualified_table_name=bq_context.fully_qualify_table_name(
                full_res_table_name
            ),
            analysis_windows=time_limits.analysis_windows,
        )

    def build_targets_query(
        self,
        time_limits: TimeLimits,
        target_list: Segment | None = None,
        custom_targets_query: str | None = None,
    ) -> str:
        return """
        {targets_query}
        """.format(
            targets_query=custom_targets_query
            or self._build_targets_query(target_list, time_limits)
        )

    def build_metrics_query(
        self,
        metric_list: list[Metric],
        time_limits: TimeLimits,
        targets_table: str,
    ) -> str:
        """Return a SQL query for querying metric data.

        For interactive use, prefer :meth:`.get_single_window_data`
        or TODO :meth:`.get_time_series_data`, according to your use case,
        which will run the query for you and return a materialized
        dataframe.

        Args:
            metric_list (list of mozanalysis.metric.Metric):
                The metrics to analyze.
            time_limits (TimeLimits): An object describing the
                interval(s) to query
            targets_table (str): The name of the targets BigQuery table

        Returns:
            A string containing a BigQuery SQL expression.

        Building this query is the main goal of this module.
        """
        analysis_windows_query = self._build_analysis_windows_query(
            time_limits.analysis_windows
        )

        metrics_columns, metrics_joins = self._build_metrics_query_bits(
            metric_list, time_limits
        )

        return """
        WITH analysis_windows AS (
            {analysis_windows_query}
        ),
        targets AS (
            SELECT
                t.*,
                aw.*
            FROM `{targets_table}` t
            CROSS JOIN analysis_windows aw
        )
        SELECT
            targets.*,
            {metrics_columns}
        FROM targets
        {metrics_joins}
        """.format(
            analysis_windows_query=analysis_windows_query,
            metrics_columns=",\n        ".join(metrics_columns),
            metrics_joins="\n".join(metrics_joins),
            targets_table=targets_table,
        )

    @staticmethod
    def _build_analysis_windows_query(analysis_windows: tuple[AnalysisWindow]) -> str:
        """Return SQL to construct a table of analysis windows.

        To query a time series, we construct a table of analysis windows
        and cross join it with the targets table to get one row per
        pair of client and analysis window.

        This method writes the SQL to define the analysis window table.
        """
        return "\n        UNION ALL\n        ".join(
            f"""(SELECT {aw.start} AS analysis_window_start,
                {aw.end} AS analysis_window_end)"""
            for aw in analysis_windows
        )

    def _partition_by_data_source(
        self, metric_list: list[Metric]
    ) -> dict[DataSource | SegmentDataSource, list[Metric | Segment]]:
        """Return a dict mapping data sources to target/metric lists."""
        data_sources = {m.data_source for m in metric_list}

        return {
            ds: [m for m in metric_list if m.data_source == ds] for ds in data_sources
        }

    def _build_targets_query(
        self, target_list: list[Segment], time_limits: TimeLimits
    ) -> str:
        target_queries = []
        target_columns = []
        dates_columns = []
        target_joins = []

        for i, t in enumerate(target_list):
            query_for_target = t.data_source.build_query_target(
                t, time_limits, from_expr_dataset=self.app_id
            )

            target_queries.append(
                f"""
        ds_{i} AS (
                {query_for_target}
            ),"""
            )

            target_columns.append(
                f"""
                    ,ds_{i}.{t.name}
                    ,ds_{i}.target_first_date as {t.name}_first_date
                """
            )

            if i != 0:
                target_joins.append(
                    f"""
                    INNER JOIN ds_{i}
                        ON ds_{i}.client_id = ds_0.client_id
                        AND ds_{i}.target_first_date <= ds_0.target_last_date
                        AND ds_{i}.target_last_date >= ds_0.target_first_date
                        """
                )

            dates_columns.append(f"{t.name}_first_date")

        target_def = "WITH" + " ".join(q for q in target_queries)

        joined_query = """
        joined AS (
            SELECT
                ds_0.client_id
                {target_columns}
            FROM
                ds_0
                {target_joins}
            ),""".format(
            target_columns=" ".join(c for c in target_columns),
            target_joins=" ".join(j for j in target_joins),
        )

        unpivot_join = """
         unpivoted AS (
                SELECT * FROM joined
                UNPIVOT(min_dates for target_date in ({target_first_dates}))
            )
        """.format(target_first_dates=", ".join(c for c in dates_columns))

        return f"""
        {target_def}
        {joined_query}
        {unpivot_join}
        SELECT
            client_id,
            max(min_dates) as enrollment_date
        FROM unpivoted
        GROUP BY client_id
        """

    def _build_metrics_query_bits(
        self, metric_list: list[Metric], time_limits: TimeLimits
    ) -> tuple[list[str], list[str]]:
        """Return lists of SQL fragments corresponding to metrics."""
        ds_metrics = self._partition_by_data_source(metric_list)
        ds_metrics = dict(ds_metrics.items())

        metrics_columns = []
        metrics_joins = []

        for i, ds in enumerate(ds_metrics.keys()):
            query_for_metrics = ds.build_query_targets(
                ds_metrics[ds],
                time_limits,
                self.experiment_name,
                analysis_length=self.analysis_length,
                continuous_enrollment=self.continuous_enrollment,
                from_expr_dataset=self.app_id,
            )
            metrics_joins.append(
                f"""    LEFT JOIN (
        {query_for_metrics}
        ) ds_{i} USING (client_id, analysis_window_start, analysis_window_end)
                """
            )

            for m in ds_metrics[ds]:
                metrics_columns.append(f"ds_{i}.{m.name}")

        return metrics_columns, metrics_joins

    def get_targets_sql(self):
        if not self._targets_sql:
            raise ValueError(
                "Target SQL not available; call `get_single_window_data` first"
            )
        return self._targets_sql

    def get_metrics_sql(self):
        if not self._metrics_sql:
            raise ValueError(
                "Metric SQL not available; call `get_single_window_data` first"
            )
        return self._metrics_sql


@attr.s(frozen=True, slots=True)
class ContinuousEnrollmentTimeLimits:
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

    analysis_windows = attr.ib(type=tuple[AnalysisWindow])

    @classmethod
    def for_single_analysis_window(
        cls,
        first_date_full_data: str,
        analysis_length_dates: str,
    ) -> ContinuousEnrollmentTimeLimits:
        """Return a ``ContinuousEnrollmentTimeLimits`` instance with the following
        parameters:

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
        analysis_window = AnalysisWindow(0, analysis_length_dates - 1)

        last_date_data_required = add_days(first_date_full_data, analysis_length_dates)

        tl = cls(
            first_enrollment_date=first_date_full_data,
            last_enrollment_date=last_date_data_required,
            first_date_data_required=first_date_full_data,
            last_date_data_required=last_date_data_required,
            analysis_windows=(analysis_window,),
        )
        return tl
