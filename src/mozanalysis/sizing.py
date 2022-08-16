# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr

from mozanalysis.bq import sanitize_table_name_for_bq
from mozanalysis.utils import hash_ish
from mozanalysis.experiment import TimeLimits


@attr.s(frozen=False, slots=True)
class HistoricalTarget:
    """Query historical data.

    The methods here query data in a way compatible with the following
    principles, which are important for experiment analysis:

    Example usage (in a colab notebook):

        from google.colab import auth
        auth.authenticate_user()
        print('Authenticated')

        from mozanalysis.sizing import HistoricalTarget
        from mozanalysis.bq import BigQueryContext
        from mozanalysis.metrics.desktop import active_hours, uri_count
        from mozanalysis.segments.desktop import new_unique_profiles, \
            new_or_resurrected_v3

        bq_context = BigQueryContext(
            dataset_id='mbowerman',  # e.g. mine's mbowerman
            project_id='moz-fx-data-bq-data-science'  # this is the default anyway
        )

        ht = HistoricalTarget(
            experiment_name='my_test_name',
            start_date='2021-01-01',
            num_dates_enrollment=8
        )

        # Run the query and get the results as a DataFrame
        res = ht.get_single_window_data(
            bq_context,
            metric_list = [
                active_hours,
                uri_count
            ],
            last_date_full_data='2021-01-30',
            analysis_start_days=0,
            analysis_length_days=21,
            target_list = [
                new_unique_profiles,
                new_or_resurrected_v3
            ]
        )

    Args:
        experiment_name (str): A simple name of this analysis, which is used
            to create table names when saving results in BigQuery.
        start_date (str): e.g. '2019-01-01'. First date for historical data to be
            retrieved.
        num_dates_enrollment (int, optional): Only include this many dates
            of dummy enrollments, where clients that satisfy the target
            criteria are added to the analysis's dataset.
            If ``None`` then use the maximum number of dates
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        analysis_length (int, optional): Number of days to include for analysis
        targets_sql (str, optional): The SQL query that was executed to create
            the targets
            table; only defined after get_single_window_data method is called.
        metrics_sql (str, optional): The SQL query that was executed to create the final
            results table; only defined after get_single_wondow_data method is
            called.


    Attributes:
        experiment_name (str): Name of the study, used in naming tables
            related to this analysis.
        start_date (str): e.g. '2019-01-01'. First date on which enrollment
            events were received.
        num_dates_enrollment (int, optional): Only include this many days
            of enrollments. If ``None`` then use the maximum number of days
            as determined by the metric's analysis window and
            ``last_date_full_data``. Typically ``7n+1``, e.g. ``8``. The
            factor '7' removes weekly seasonality, and the ``+1`` accounts
            for the fact that enrollment typically starts a few hours
            before UTC midnight.
        analysis_length (int, optional): Number of days to include for analysis
        targets_sql (str): The SQL query that was executed to create the targets
            table; only defined after get_single_window_data method is called.
        metrics_sql (str): The SQL query that was executed to create the final
            results table; only defined after get_single_wondow_data method is
            called.

    """

    experiment_name = attr.ib()
    start_date = attr.ib()
    num_dates_enrollment = attr.ib(default=None)
    analysis_length = attr.ib(default=None)
    targets_sql = attr.ib(default='')
    metrics_sql = attr.ib(default='')

    def get_single_window_data(
        self,
        bq_context,
        metric_list,
        last_date_full_data,
        analysis_start_days,
        analysis_length_days,
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
            last_date_full_data (str): The most recent date for which we
                have complete data, e.g. '2019-03-22'. If you want to ignore
                all data collected after a certain date (e.g. when the
                experiment recipe was deactivated), then do that here.
            analysis_start_days (int): the start of the analysis window,
                measured in 'days since the client enrolled'. We ignore data
                collected outside this analysis window.
            analysis_length_days (int): the length of the analysis window,
                measured in days.
            target_list (list of mozanalysis.segments.Segment): The targets
                that define clients to be included in the analysis, based on
                inclusion in a defined user segment.
            custom_targets_query (str): A full SQL query to be used
                in the main query::

                    WITH targets AS ({custom_targets_query})

                N.B. this query's results must be uniquely keyed by
                (client_id), or else your results will be subtly
                wrong.

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

        time_limits = TimeLimits.for_single_analysis_window(
            self.start_date,
            last_date_full_data,
            analysis_start_days,
            analysis_length_days,
            self.num_dates_enrollment,
        )

        self.targets_sql = self.build_targets_query(
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
                    hash_ish(self.targets_sql),
                ]
            )
        )

        bq_context.run_query(self.targets_sql, targets_table_name)

        self.metrics_sql = self.build_metrics_query(
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
                    hash_ish(self.metrics_sql)
                ]
            )
        )

        return bq_context.run_query(
            self.metrics_sql,
            full_res_table_name).to_dataframe()

    def build_targets_query(
        self,
        time_limits,
        target_list=None,
        custom_targets_query=None
    ):

        return """
        WITH targets AS (
            {targets_query}
        )
        SELECT
            t.client_id
        FROM targets t
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

    def _partition_by_data_source(self, metric_or_target_list):
        """Return a dict mapping data sources to target/metric lists."""
        data_sources = {m.data_source for m in metric_or_target_list}

        return {
            ds: [m for m in metric_or_target_list if m.data_source == ds]
            for ds in data_sources
        }

    def _build_targets_query(self, target_list, time_limits):
        """Return lists of SQL fragments corresponding to targets."""
        ds_targets = self._partition_by_data_source(target_list)

        for i, ds in enumerate(ds_targets.keys()):
            query_for_targets = ds.build_query_target(
                ds_targets[ds], time_limits
            )
            if i == 0:
                targets_join = """SELECT
                    ds_0.client_id
                FROM (
                    {query}
                ) ds_0
                """.format(query=query_for_targets)

                if len(target_list) == 1:
                    return targets_join
            else:
                targets_join += """  LEFT JOIN (
            {query}
            ) ds_{i} USING (client_id)
                    """.format(
                        query=query_for_targets, i=i
                    )
        return targets_join

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
