# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from textwrap import dedent

from mozanalysis.metrics import DataSource, Metric, agg_any, agg_sum

#: DataSource: The clients_daily table.
clients_daily = DataSource(
    name="clients_daily",
    from_expr="mozdata.telemetry.clients_daily",
)

#: DataSource: The `search_clients_engines_sources_daily`_ table.
#: This table unpacks search counts from the main ping;
#: it contains one row per (client_id, submission_date, engine, source).
#:
#: .. _`search_clients_engines_sources_daily`: https://docs.telemetry.mozilla.org/
#:    datasets/search/search_clients_engines_sources_daily/reference.html
search_clients_engines_sources_daily = DataSource(
    name="search_clients_engines_sources_daily",
    from_expr="mozdata.search.search_clients_engines_sources_daily",
    experiments_column_type=None,
)

#: DataSource: A clone of `search_clients_engines_sources_daily`.
#: Exists for backwards compatibility; new uses should use the new name.
search_clients_daily = search_clients_engines_sources_daily

#: DataSource: The main_summary table.
main_summary = DataSource(
    name="main_summary", from_expr="mozdata.telemetry.main_summary"
)

#: DataSource: The events table.
events = DataSource(
    name="events",
    from_expr="mozdata.telemetry.events",
    experiments_column_type="native",
)

#: DataSource: Normandy events; a subset of the events table.
#: The telemetry.events table is clustered by event_category.
#: Normandy accounts for about 10% of event volume, so this dramatically
#: reduces bytes queried compared to counting rows from the generic events DataSource.
normandy_events = DataSource(
    name="normandy_events",
    from_expr="""(
        SELECT
            *
        FROM mozdata.telemetry.events
        WHERE event_category = 'normandy'
    )""",
    experiments_column_type="native",
)

#: DataSource: The telemetry_stable.main_v4 ping table.
#: The main_v4 table is what backs the telemetry.main view.
#: Referencing the table directly helps us stay under the BigQuery
#: query complexity budget.
main = DataSource(
    name="main",
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date,
                    environment.experiments
                FROM `moz-fx-data-shared-prod`.telemetry_stable.main_v4
            )""",
    experiments_column_type="native",
)

#: DataSource: The telemetry.crash ping table.
crash = DataSource(
    name="crash",
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date,
                    environment.experiments
                FROM mozdata.telemetry.crash
            )""",
    experiments_column_type="native",
)

#: DataSource: The ``messaging_system.cfr`` table.
cfr = DataSource(
    name="cfr",
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date
                FROM `moz-fx-data-derived-datasets`.messaging_system.cfr
            )""",
    experiments_column_type="native",
)

#: DataSource: The ``activity_stream.events`` table.
activity_stream_events = DataSource(
    name="activity_stream_events",
    from_expr="""(
                SELECT
                    *,
                    DATE(submission_timestamp) AS submission_date
                FROM mozdata.activity_stream.events
            )""",
    experiments_column_type="native",
)


#: Metric: ...
active_hours = Metric(
    name="active_hours",
    data_source=clients_daily,
    select_expr=agg_sum("active_hours_sum"),
    friendly_name="Active hours",
    description=dedent(
        """\
        Measures the amount of time (in 5-second increments) during which
        Firefox received user input from a keyboard or mouse. The Firefox
        window does not need to be focused.
    """
    ),
)

#: Metric: ...
uri_count = Metric(
    name="uri_count",
    data_source=clients_daily,
    select_expr=agg_sum("scalar_parent_browser_engagement_total_uri_count_sum"),
    friendly_name="URIs visited",
    description=dedent(
        """\
        Counts the total number of URIs visited.
        Includes within-page navigation events (e.g. to anchors).
    """
    ),
)

#: Metric: ...
search_count = Metric(
    name="search_count",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("sap"),
    friendly_name="SAP searches",
    description=dedent(
        """\
        Counts the number of searches a user performed through Firefox's
        Search Access Points.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """  # noqa:E501
    ),
)

#: Metric: ...
tagged_search_count = Metric(
    name="tagged_search_count",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("tagged_sap"),
    friendly_name="Tagged SAP searches",
    description=dedent(
        """\
        Counts the number of searches a user performed through Firefox's
        Search Access Points that were submitted with a partner code
        and were potentially revenue-generating.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """  # noqa:E501
    ),
)

#: Metric: ...
tagged_follow_on_search_count = Metric(
    name="tagged_follow_on_search_count",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("tagged_follow_on"),
    friendly_name="Tagged follow-on searches",
    description=dedent(
        """\
        Counts the number of follow-on searches with a Mozilla partner tag.
        These are additional searches that users performed from a search engine
        results page after executing a tagged search through a SAP.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """  # noqa:E501
    ),
)

#: Metric: ...
ad_clicks = Metric(
    name="ad_clicks",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("ad_click"),
    friendly_name="Ad clicks",
    description=dedent(
        """\
        Counts clicks on ads on search engine result pages with a Mozilla
        partner tag.
    """
    ),
)

#: Metric: ...
searches_with_ads = Metric(
    name="searches_with_ads",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("search_with_ads"),
    friendly_name="Search result pages with ads",
    description=dedent(
        """\
        Counts search result pages served with advertising.
        Users may not actually see these ads thanks to e.g. ad-blockers.
        Learn more in the
        [search analysis documentation](https://mozilla-private.report/search-analysis-docs/book/in_content_searches.html).
    """  # noqa:E501
    ),
)

#: Metric: ...
organic_search_count = Metric(
    name="organic_search_count",
    data_source=search_clients_engines_sources_daily,
    select_expr=agg_sum("organic"),
    friendly_name="Organic searches",
    description=dedent(
        """\
        Counts organic searches, which are searches that are _not_ performed
        through a Firefox SAP and which are not monetizable.
        Learn more in the
        [search data documentation](https://docs.telemetry.mozilla.org/datasets/search.html).
    """  # noqa:E501
    ),
)

#: Metric: ...
unenroll = Metric(
    name="unenroll",
    data_source=normandy_events,
    select_expr=agg_any(
        """
                event_category = 'normandy'
                AND event_method = 'unenroll'
                AND event_string_value = '{experiment_slug}'
            """
    ),
    friendly_name="Unenrollments",
    description=dedent(
        """\
        Counts the number of clients with an experiment unenrollment event.
    """
    ),
    bigger_is_better=False,
)

#: Metric: ...
view_about_logins = Metric(
    name="view_about_logins",
    data_source=events,
    select_expr=agg_any(
        """
                event_method = 'open_management'
                AND event_category = 'pwmgr'
            """
    ),
    friendly_name="about:logins viewers",
    description=dedent(
        """\
        Counts the number of clients that viewed about:logins.
    """
    ),
)

#: Metric: ...
view_about_protections = Metric(
    name="view_about_protections",
    data_source=events,
    select_expr=agg_any(
        """
                event_method = 'show'
                AND event_object = 'protection_report'
            """
    ),
    friendly_name="about:protections viewers",
    description=dedent(
        """\
        Counts the number of clients that viewed about:protections.
    """
    ),
)

#: Metric: ...
connect_fxa = Metric(
    name="connect_fxa",
    data_source=events,
    select_expr=agg_any(
        """
                event_method = 'connect'
                AND event_object = 'account'
            """
    ),
    friendly_name="Connected FxA",
    description=dedent(
        """\
        Counts the number of clients that took action to connect to FxA.
        This does not include clients that were already connected to FxA at
        the start of the experiment and remained connected.
    """
    ),
)

#: Metric: Pocket organic rec clicks in New Tab
pocket_rec_clicks = Metric(
    name="pocket_rec_clicks",
    data_source=activity_stream_events,
    select_expr="""COUNTIF(
                event = 'CLICK'
                AND source = 'CARDGRID'
                AND JSON_EXTRACT_SCALAR(value, '$.card_type') = 'organic'
            )""",
    friendly_name="Clicked Pocket organic recs in New Tab",
    description=dedent(
        """\
         Counts the number of Pocket rec clicks made by each client.
    """
    ),
)

#: Metric: Pocket sponsored content clicks in New Tab
pocket_spoc_clicks = Metric(
    name="pocket_spoc_clicks",
    data_source=activity_stream_events,
    select_expr="""COUNTIF(
                event = 'CLICK'
                AND source = 'CARDGRID'
                AND JSON_EXTRACT_SCALAR(value, '$.card_type') = 'spoc'
            )""",
    friendly_name="Clicked Pocket sponsored content in New Tab",
    description=dedent(
        """\
         Counts the number of Pocket sponsored content clicks made by each client.
    """
    ),
)

#: Metric: ...
days_of_use = Metric(
    name="days_of_use",
    data_source=clients_daily,
    select_expr="COUNT(ds.submission_date)",
    friendly_name="Days of use",
    description="The number of days in the interval that each client sent a main ping.",
)

#: Metric: Clicks to disable Pocket in New Tab
disable_pocket_clicks = Metric(
    name="disable_pocket_clicks",
    data_source=activity_stream_events,
    select_expr="""COUNTIF(
                event = 'PREF_CHANGED'
                AND source = 'TOP_STORIES'
                AND JSON_EXTRACT_SCALAR(value, '$.status') = 'false'
            )""",
    friendly_name="Disabled Pocket in New Tab",
    description=dedent(
        """\
         Counts the number of clicks to disable Pocket in New Tab made by each client.
    """
    ),
)

#: Metric: Clicks to disable Pocket sponsored content in New Tab
disable_pocket_spocs_clicks = Metric(
    name="disable_pocket_spocs_clicks",
    data_source=activity_stream_events,
    select_expr="""COUNTIF(
                event = 'PREF_CHANGED'
                AND source = 'POCKET_SPOCS'
                AND JSON_EXTRACT_SCALAR(value, '$.status') = 'false'
            )""",
    friendly_name="Disabled Pocket sponsored content in New Tab",
    description=dedent(
        """\
         Counts the number of clicks to disable Pocket sponsored content
         in New Tab made by each client.
    """
    ),
)
