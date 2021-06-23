# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.metrics import DataSource, Metric, agg_sum

#: DataSource: The baseline ping table.
baseline = DataSource(
    name="baseline",
    from_expr="""(
                SELECT
                    p.*,
                    DATE(p.submission_timestamp) AS submission_date
                FROM `moz-fx-data-shared-prod.{dataset}.baseline` p
            )""",
    client_id_column="client_info.client_id",
    experiments_column_type="glean",
    default_dataset="org_mozilla_firefox",
)


#: DataSource: Events table.
#: For convenience, this is exploded to one-row-per-event
#: like the ``telemetry.events`` dataset.
events = DataSource(
    name="events",
    from_expr="""(
                SELECT
                    p.* EXCEPT (events),
                    DATE(p.submission_timestamp) AS submission_date,
                    event
                FROM
                    `moz-fx-data-shared-prod.{dataset}.events` p
                CROSS JOIN
                    UNNEST(p.events) AS event
            )""",
    client_id_column="client_info.client_id",
    experiments_column_type="glean",
    default_dataset="org_mozilla_firefox",
)


#: DataSource: Metrics ping table.
metrics = DataSource(
    name="metrics",
    from_expr="""(
                SELECT
                    p.*,
                    DATE(p.submission_timestamp) AS submission_date
                FROM `moz-fx-data-shared-prod.{dataset}.metrics` p
            )""",
    client_id_column="client_info.client_id",
    experiments_column_type="glean",
    default_dataset="org_mozilla_firefox",
)


#: Metric: ...
uri_count = Metric(
    name="uri_count",
    data_source=baseline,
    select_expr=agg_sum("metrics.counter.events_total_uri_count"),
    friendly_name="URIs visited",
    description="Counts the number of URIs each client visited",
)

#: Metric: ...
user_reports_site_issue_count = Metric(
    name="user_reports_site_issue_count",
    data_source=events,
    select_expr="COUNTIF(event.name = 'browser_menu_action' AND "
    + "mozfun.map.get_key('event.extra', 'item') = 'report_site_issue')",
    friendly_name="Site issues reported",
    description="Counts the number of times clients reported an issue with a site.",
)

#: Metric: ...
user_reload_count = Metric(
    name="user_reload_count",
    data_source=events,
    select_expr="COUNTIF(event.name = 'browser_menu_action' AND "
    + "mozfun.map.get_key('event.extra', 'item') = 'reload')",
    friendly_name="Pages reloaded",
    description="Counts the number of times a client reloaded a page.",
    bigger_is_better=False,
)

#: Metric: ...
baseline_ping_count = Metric(
    name="baseline_ping_count",
    data_source=baseline,
    select_expr="COUNT(document_id)",
    friendly_name="Baseline pings",
    description="Counts the number of `baseline` pings received from each client.",
)

#: Metric: ...
metric_ping_count = Metric(
    name="metric_ping_count",
    data_source=metrics,
    select_expr="COUNT(document_id)",
    friendly_name="Metrics pings",
    description="Counts the number of `metrics` pings received from each client.",
)

#: Metric: ...
first_run_date = Metric(
    name="first_run_date",
    data_source=baseline,
    select_expr="MIN(client_info.first_run_date)",
    friendly_name="First run date",
    description="The earliest first-run date reported by each client.",
)
