# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.metrics import Metric, DataSource, agg_sum


baseline = DataSource(
    name='baseline',
    from_expr="""(
        SELECT
            p.*,
            p.client_info.client_id AS client_id,
            DATE(p.submission_timestamp) AS submission_date
        FROM `moz-fx-data-shared-prod.org_mozilla_fenix.baseline` p
    )"""
)


events = DataSource(
    name='events',
    from_expr="""(
        SELECT
            p.* EXCEPT (events),
            p.client_info.client_id AS client_id,
            DATE(p.submission_timestamp) AS submission_date
            event
        FROM
            `moz-fx-data-shared-prod.org_mozilla_fenix.events` p
        CROSS JOIN
            UNNEST(p.events) AS event
    )"""
)


metrics = DataSource(
    name='metrics',
    from_expr="""(
        SELECT
            p.*,
            p.client_info.client_id AS client_id,
            DATE(p.submission_timestamp) AS submission_date
        FROM `moz-fx-data-shared-prod.org_mozilla_fenix.metrics` p
    )"""
)


uri_count = Metric(
    name='uri_count',
    data_source=baseline,
    select_expr=agg_sum('counter.events_total_uri_count')
)

# # TODO: work out how to deal with the potentially varying units
# duration = Metric(
#     name='duration',
#     data_source=baseline,
#     select_expr=...
# )

user_reports_site_issue_count = Metric(
    name='user_reports_site_issue_count',
    data_source=events,
    select_expr=agg_sum(
        "event.name = 'browser_menu_action'"
        " AND event.key = 'item'"
        " AND event.value = 'report_site_issue'"
    )
)

user_reload_count = Metric(
    name='user_reload_count',
    data_source=events,
    select_expr=agg_sum(
        "event.name = 'browser_menu_action'"
        " AND event.key = 'item'"
        " AND event.value = 'reload'"
    )
)

baseline_ping_count = Metric(
    name='baseline_ping_count',
    data_source=baseline,
    select_expr='COUNT(ds.client_id)'
)

metric_ping_count = Metric(
    name='metric_ping_count',
    data_source=metrics,
    select_expr='COUNT(ds.client_id)'
)

first_run_date = Metric(
    name='first_run_date',
    data_source=baseline,
    select_expr='MIN(first_run_date)'
)
