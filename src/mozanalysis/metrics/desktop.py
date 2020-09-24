# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from textwrap import dedent

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

# The telemetry.events table is clustered by event_category.
# Normandy accounts for about 10% of event volume, so this dramatically
# reduces bytes queried compared to counting rows from the generic events DataSource.
normandy_events = DataSource(
    name='normandy_events',
    from_expr="""(
        SELECT
            *
        FROM `moz-fx-data-shared-prod`.telemetry.events
        WHERE event_category = 'normandy'
    )""",
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
    select_expr=agg_sum('active_hours_sum'),
    friendly_name="Active hours",
    description=dedent("""\
        Measures the amount of time (in 5-second increments) during which
        Firefox received user input from a keyboard or mouse. The Firefox
        window does not need to be focused.
    """),
)

uri_count = Metric(
    name='uri_count',
    data_source=clients_daily,
    select_expr=agg_sum('scalar_parent_browser_engagement_total_uri_count_sum'),
    friendly_name="URIs visited",
    description=dedent("""\
        Counts the total number of URIs visited.
        Includes within-page navigation events (e.g. to anchors).
    """),
)

search_count = Metric(
    name='search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('sap'),
    friendly_name="SAP searches",
    description=dedent("""\
        Counts the number of searches a user performed through Firefox's
        Search Access Points.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """),  # noqa:E501
)

tagged_search_count = Metric(
    name='tagged_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('tagged_sap'),
    friendly_name="Tagged SAP searches",
    description=dedent("""\
        Counts the number of searches a user performed through Firefox's
        Search Access Points that were submitted with a partner code
        and were potentially revenue-generating.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """),  # noqa:E501
)

tagged_follow_on_search_count = Metric(
    name='tagged_follow_on_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('tagged_follow_on'),
    friendly_name="Tagged follow-on searches",
    description=dedent("""\
        Counts the number of follow-on searches with a Mozilla partner tag.
        These are additional searches that users performed from a search engine
        results page after executing a tagged search through a SAP.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """),  # noqa:E501
)

ad_clicks = Metric(
    name='ad_clicks',
    data_source=search_clients_daily,
    select_expr=agg_sum('ad_click'),
    friendly_name="Ad clicks",
    description=dedent("""\
        Counts clicks on ads on search engine result pages with a Mozilla
        partner tag.
    """),
)

searches_with_ads = Metric(
    name='searches_with_ads',
    data_source=search_clients_daily,
    select_expr=agg_sum('search_with_ads'),
    friendly_name="Search result pages with ads",
    description=dedent("""\
        Counts search result pages served with advertising.
        Users may not actually see these ads thanks to e.g. ad-blockers.
        Learn more in the
        [search analysis documentation](https://mozilla-private.report/search-analysis-docs/book/in_content_searches.html).
    """),  # noqa:E501
)

organic_search_count = Metric(
    name='organic_search_count',
    data_source=search_clients_daily,
    select_expr=agg_sum('organic'),
    friendly_name="Organic searches",
    description=dedent("""\
        Counts organic searches, which are searches that are _not_ performed
        through a Firefox SAP and which are not monetizable.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """),  # noqa:E501
)

unenroll = Metric(
    name='unenroll',
    data_source=normandy_events,
    select_expr=agg_any("""
                event_category = 'normandy'
                AND event_method = 'unenroll'
                AND event_string_value = '{experiment_slug}'
            """),
    friendly_name="Unenrollments",
    description=dedent("""\
        Counts the number of clients with an experiment unenrollment event.
    """),
)

view_about_logins = Metric(
    name='view_about_logins',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'open_management'
                AND event_category = 'pwmgr'
            """),
    friendly_name="about:logins viewers",
    description=dedent("""\
        Counts the number of clients that viewed about:logins.
    """),
)

view_about_protections = Metric(
    name='view_about_protections',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'show'
                AND event_object = 'protection_report'
            """),
    friendly_name="about:protections viewers",
    description=dedent("""\
        Counts the number of clients that viewed about:protections.
    """),
)

connect_fxa = Metric(
    name='connect_fxa',
    data_source=events,
    select_expr=agg_any("""
                event_method = 'connect'
                AND event_object = 'account'
            """),
    friendly_name="Connected FxA",
    description=dedent("""\
        Counts the number of clients that took action to connect to FxA.
        This does not include clients that were already connected to FxA at
        the start of the experiment and remained connected.
    """),
)
