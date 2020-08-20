# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.metrics import Metric, DataSource, agg_sum


baseline = DataSource(
    name='baseline',
    from_expr="""(
                SELECT
                    p.*,
                    DATE(p.submission_timestamp) AS submission_date
                FROM `moz-fx-data-shared-prod.org_mozilla_fenix.baseline` p
            )""",
    client_id_column='client_info.client_id',
    experiments_column_type='glean',
)


events = DataSource(
    name='events',
    from_expr="""(
                SELECT
                    p.* EXCEPT (events),
                    DATE(p.submission_timestamp) AS submission_date,
                    event
                FROM
                    `moz-fx-data-shared-prod.org_mozilla_fenix.events` p
                CROSS JOIN
                    UNNEST(p.events) AS event
            )""",
    client_id_column='client_info.client_id',
    experiments_column_type='glean',
)


metrics = DataSource(
    name='metrics',
    from_expr="""(
                SELECT
                    p.*,
                    DATE(p.submission_timestamp) AS submission_date
                FROM `moz-fx-data-shared-prod.org_mozilla_fenix.metrics` p
            )""",
    client_id_column='client_info.client_id',
    experiments_column_type='glean',
)


uri_count = Metric(
    name='uri_count',
    data_source=baseline,
    select_expr=agg_sum('metrics.counter.events_total_uri_count')
)

user_reports_site_issue_count = Metric(
    name='user_reports_site_issue_count',
    data_source=events,
    select_expr="COUNTIF(event.name = 'browser_menu_action' AND "
    + "mozfun.map.get_key('event.extra', 'item') = 'report_site_issue')"
)

user_reload_count = Metric(
    name='user_reload_count',
    data_source=events,
    select_expr="COUNTIF(event.name = 'browser_menu_action' AND "
    + "mozfun.map.get_key('event.extra', 'item') = 'reload')"
)

baseline_ping_count = Metric(
    name='baseline_ping_count',
    data_source=baseline,
    select_expr='COUNT(client_info.client_id)'
)

metric_ping_count = Metric(
    name='metric_ping_count',
    data_source=metrics,
    select_expr='COUNT(client_info.client_id)'
)

first_run_date = Metric(
    name='first_run_date',
    data_source=baseline,
    select_expr='MIN(client_info.first_run_date)'
)
