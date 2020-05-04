# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.metrics import Metric, DataSource, agg_sum, agg_any


clients_daily = DataSource(
    name='clients_daily',
    from_expr="`moz-fx-data-shared-prod.telemetry.clients_daily`",
)

search_clients_daily = DataSource(
    name='search_clients_daily',
    from_expr='`moz-fx-data-shared-prod.search.search_clients_daily`',
    experiments_column_type=None,
)

main_summary = DataSource(
    name='main_summary',
    from_expr="`moz-fx-data-shared-prod.telemetry.main_summary`"
)

events = DataSource(
    name='events',
    from_expr="`moz-fx-data-shared-prod.telemetry.events`",
    experiments_column_type='native',
)

main = DataSource(
    name='main',
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date,
                    environment.experiments
                FROM `moz-fx-data-shared-prod`.telemetry.main
            )""",
    experiments_column_type="native",
)

crash = DataSource(
    name='crash',
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date,
                    environment.experiments
                FROM `moz-fx-data-shared-prod`.telemetry.crash
            )""",
    experiments_column_type="native",
)

cfr = DataSource(
    name='cfr',
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date
                FROM `moz-fx-data-derived-datasets`.messaging_system.cfr
            )""",
    experiments_column_type="native",
)

active_hours = Metric(
    name='active_hours',
    data_source=clients_daily,
    select_expr=agg_sum('active_hours_sum')
)

uri_count = Metric(
    name='uri_count',
    data_source=clients_daily,
    select_expr=agg_sum('scalar_parent_browser_engagement_total_uri_count_sum')
)

search_count = Metric(
    name='search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('sap')
)

tagged_search_count = Metric(
    name='tagged_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('tagged_sap')
)

tagged_follow_on_search_count = Metric(
    name='tagged_follow_on_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('tagged_follow_on')
)

ad_clicks = Metric(
    name='ad_clicks',
    data_source=search_clients_daily,
    select_expr=agg_sum('ad_click')
)

searches_with_ads = Metric(
    name='searches_with_ads',
    data_source=search_clients_daily,
    select_expr=agg_sum('search_with_ads')
)

organic_search_count = Metric(
    name='organic_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('organic')
)

unenroll = Metric(
    name='unenroll',
    data_source=events,
    select_expr=agg_any("""
                event_category = 'normandy'
                AND event_method = 'unenroll'
                AND event_string_value = '{experiment_slug}'
            """)
)

view_about_logins = Metric(
    name='view_about_logins',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'open_management'
                AND event_category = 'pwmgr'
            """)
)

view_about_protections = Metric(
    name='view_about_protections',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'show'
                AND event_object = 'protection_report'
            """)
)

connect_fxa = Metric(
    name='connect_fxa',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'connect'
                AND event_object = 'account'
            """)
)
