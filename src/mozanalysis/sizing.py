# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr

from mozanalysis.bq import sanitize_table_name_for_bq
from mozanalysis.utils import hash_ish
from mozanalysis.experiment import TimeLimits, TimeSeriesResult, add_days
from datetime import date


@attr.s(frozen=False, slots=True)
class HistoricalTarget:
    """Query historical data.

    The methods here query data in a way compatible with the following
    principles, which are important for experiment analysis:

    Example usage (in a colab notebook):

        from google.colab import auth
        auth.authenticate_user()
        print('Authenticated')
        # Or, if running in a local notebook, authorize with gcloud CLI:
        # `gcloud auth login --update-adc`


        from mozanalysis.metrics.desktop import active_hours, uri_count
        from mozanalysis.segments.desktop import allweek_regular_v1, \
            new_or_resurrected_v3

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

        # (Optional) After invoking `get_single_window_data`, return the targets
        # and metrics queries used to construct the results DataFrame:
        targets_sql = ht.get_targets_sql()

        metrics_sql = ht.get_metrics_sql()

    Args:
        experiment_name (str): A simple name of this analysis, which is used
            to create table names when saving results in BigQuery.
        start_date (str): e.g. '2019-01-01'. First date for historical data to be
            retrieved.
        num_dates_enrollment (int): Number of days to consider to enroll clients
            in the analysis.
        analysis_length (int, optional): Number of days to include for analysis

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
    num_dates_enrollment = attr.ib()
    analysis_length = attr.ib(default=None)
    _metrics_sql = attr.ib(default=None)
    _targets_sql = attr.ib(default=None)

    def get_single_window_data(
        self,
        bq_context,
        metric_list,
        target_list=None,
        custom_targets_query=None
    ):
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

        last_date_full_data = add_days(
            self.start_date,
            self.num_dates_enrollment + self.analysis_length - 1
        )

        today = date.today().strftime('%Y-%m-%d')
        if last_date_full_data >= today:
            raise ValueError(
                "Based on the start date, {}".format(
                    self.start_date
                )
                + ", with {} days of enrollment ".format(
                    self.num_dates_enrollment
                )
                + "and analysis of length {} days, ".format(
                    self.analysis_length
                )
                + "the last day of analysis is {}".format(
                    last_date_full_data
                )
                + ", which is in the future."
            )

        time_limits = TimeLimits.for_single_analysis_window(
            self.start_date,
            last_date_full_data,
            0,
            self.analysis_length,
            self.num_dates_enrollment,
        )

        self._targets_sql = self.build_targets_query(
            time_limits=time_limits,
            target_list=target_list,
            custom_targets_query=custom_targets_query
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

        bq_context.run_query(self._targets_sql, targets_table_name)

        self._metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            targets_table=bq_context.fully_qualify_table_name(
                targets_table_name
            ),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    self.experiment_name,
                    hash_ish(self._metrics_sql)
                ]
            )
        )

        return bq_context.run_query(
            self._metrics_sql,
            full_res_table_name).to_dataframe()

    def get_time_series_data(
            self,
            bq_context,
            metric_list,
            time_series_period="weekly",
            custom_targets_query=None,
            target_list=None,
    ):

        last_date_full_data = add_days(
            self.start_date,
            self.num_dates_enrollment + self.analysis_length - 1
        )

        today = date.today().strftime('%Y-%m-%d')
        if last_date_full_data >= today:
            raise ValueError(
                "Based on the start date, {}".format(
                    self.start_date
                )
                + ", with {} days of enrollment ".format(
                    self.num_dates_enrollment
                )
                + "and analysis of length {} days, ".format(
                    self.analysis_length
                )
                + "the last day of analysis is {}".format(
                    last_date_full_data
                )
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
            custom_targets_query=custom_targets_query
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

        bq_context.run_query(self._targets_sql, targets_table_name)

        self._metrics_sql = self.build_metrics_query(
            metric_list=metric_list,
            time_limits=time_limits,
            targets_table=bq_context.fully_qualify_table_name(
                targets_table_name
            ),
        )

        full_res_table_name = sanitize_table_name_for_bq(
            "_".join(
                [
                    time_limits.last_date_data_required,
                    self.experiment_name,
                    hash_ish(self._metrics_sql)
                ]
            )
        )

        bq_context.run_query(self._metrics_sql, full_res_table_name)

        return TimeSeriesResult(
            fully_qualified_table_name=bq_context.fully_qualify_table_name(
                full_res_table_name
            ),
            analysis_windows=time_limits.analysis_windows,
        )

    def build_targets_query(
        self,
        time_limits,
        target_list=None,
        custom_targets_query=None
    ):

        return """
        {targets_query}
        """.format(
            targets_query=custom_targets_query or
            self._build_targets_query(target_list, time_limits)
        )

    def build_metrics_query(
        self,
        metric_list,
        time_limits,
        targets_table,
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
            targets_table=targets_table
        )

    @staticmethod
    def _build_analysis_windows_query(analysis_windows):
        """Return SQL to construct a table of analysis windows.

        To query a time series, we construct a table of analysis windows
        and cross join it with the targets table to get one row per
        pair of client and analysis window.

        This method writes the SQL to define the analysis window table.
        """
        return "\n        UNION ALL\n        ".join(
            """(SELECT {aws} AS analysis_window_start,
                {awe} AS analysis_window_end)""".format(
                aws=aw.start,
                awe=aw.end,
            )
            for aw in analysis_windows
        )

    def _partition_by_data_source(self, metric_list):
        """Return a dict mapping data sources to target/metric lists."""
        data_sources = {m.data_source for m in metric_list}

        return {
            ds: [m for m in metric_list if m.data_source == ds]
            for ds in data_sources
        }

    def _build_targets_query(self, target_list, time_limits):

        target_queries = []
        target_columns = []
        dates_columns = []
        target_joins = []

        for i, t in enumerate(target_list):
            query_for_target = t.data_source.build_query_target(
                t, time_limits
            )

            target_queries.append(
                """
        ds_{i} AS (
                {query}
            ),""".format(i=i, query=query_for_target)
            )

            target_columns.append(
                """
                    ,ds_{i}.{name}
                    ,ds_{i}.target_first_date as {name}_first_date
                """.format(i=i, name=t.name)
            )

            if i != 0:
                target_joins.append("""
                    LEFT JOIN ds_{i}
                        ON ds_{i}.client_id = ds_0.client_id
                        AND ds_{i}.target_first_date <= ds_0.target_last_date
                        AND ds_{i}.target_last_date >= ds_0.target_first_date
                        """.format(
                            i=i
                        )
                )

            dates_columns.append(
                "{name}_first_date".format(name=t.name)
            )

        target_def = "WITH" + " ".join(
                    q for q in target_queries
                )

        joined_query = """
        joined AS (
            SELECT
                ds_0.client_id
                {target_columns}
            FROM
                ds_0
                {target_joins}
            ),""".format(target_columns=" ".join(
                    c for c in target_columns
                ),
                target_joins=" ".join(
                    j for j in target_joins
                ),
            )

        unpivot_join = """
         unpivoted AS (
                SELECT * FROM joined
                UNPIVOT(min_dates for target_date in ({target_first_dates}))
            )
        """.format(target_first_dates=", ".join(
            c for c in dates_columns
            )
        )

        return """
        {query_for_targets}
        {joined_query}
        {final_table}
        SELECT
            client_id,
            max(min_dates) as enrollment_date
        FROM unpivoted
        GROUP BY client_id
        """.format(
            query_for_targets=target_def,
            joined_query=joined_query,
            final_table=unpivot_join
        )

    def _build_metrics_query_bits(
        self,
        metric_list,
        time_limits
    ):
        """Return lists of SQL fragments corresponding to metrics."""
        ds_metrics = self._partition_by_data_source(metric_list)
        ds_metrics = {
            ds: metrics
            for ds, metrics in ds_metrics.items()
        }

        metrics_columns = []
        metrics_joins = []

        for i, ds in enumerate(ds_metrics.keys()):
            query_for_metrics = ds.build_query_targets(
                ds_metrics[ds],
                time_limits,
                self.experiment_name
            )
            metrics_joins.append(
                """    LEFT JOIN (
        {query}
        ) ds_{i} USING (client_id, analysis_window_start, analysis_window_end)
                """.format(
                    query=query_for_metrics, i=i
                )
            )

            for m in ds_metrics[ds]:
                metrics_columns.append(
                    "ds_{i}.{metric_name}".format(i=i, metric_name=m.name)
                )

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
