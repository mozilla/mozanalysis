import pytest
from cheap_lint import sql_lint
from mozanalysis.config import ConfigLoader


@pytest.fixture()
def included_metrics():
    "Returns a thorough but not necessarily exhaustive list of metrics"
    desktop_metrics = [
        ConfigLoader.get_metric("active_hours", "firefox_desktop"),
        ConfigLoader.get_metric("uri_count", "firefox_desktop"),
        ConfigLoader.get_metric("search_count", "firefox_desktop"),
        ConfigLoader.get_metric("tagged_search_count", "firefox_desktop"),
        ConfigLoader.get_metric("tagged_follow_on_search_count", "firefox_desktop"),
        ConfigLoader.get_metric("ad_clicks", "firefox_desktop"),
        ConfigLoader.get_metric("searches_with_ads", "firefox_desktop"),
        ConfigLoader.get_metric("organic_search_count", "firefox_desktop"),
        ConfigLoader.get_metric("unenroll", "firefox_desktop"),
        ConfigLoader.get_metric("view_about_logins", "firefox_desktop"),
        ConfigLoader.get_metric("view_about_protections", "firefox_desktop"),
        ConfigLoader.get_metric("connect_fxa", "firefox_desktop"),
        ConfigLoader.get_metric("pocket_rec_clicks", "firefox_desktop"),
        ConfigLoader.get_metric("pocket_spoc_clicks", "firefox_desktop"),
        ConfigLoader.get_metric("days_of_use", "firefox_desktop"),
        ConfigLoader.get_metric("qualified_cumulative_days_of_use", "firefox_desktop"),
        ConfigLoader.get_metric("disable_pocket_clicks", "firefox_desktop"),
        ConfigLoader.get_metric("disable_pocket_spocs_clicks", "firefox_desktop"),
    ]

    fenix_metrics = [
        ConfigLoader.get_metric("uri_count", "fenix"),
        ConfigLoader.get_metric("user_reports_site_issue_count", "fenix"),
        ConfigLoader.get_metric("user_reload_count", "fenix"),
        ConfigLoader.get_metric("baseline_ping_count", "fenix"),
        ConfigLoader.get_metric("metric_ping_count", "fenix"),
        ConfigLoader.get_metric("first_run_date", "fenix"),
    ]

    other_metrics = []
    for platform in [
        "firefox_ios",
        "focus_android",
        "focus_ios",
        "klar_android",
        "klar_ios",
    ]:
        other_metrics += [
            ConfigLoader.get_metric("baseline_ping_count", platform),
            ConfigLoader.get_metric("metric_ping_count", platform),
            ConfigLoader.get_metric("first_run_date", platform),
        ]

    return desktop_metrics + fenix_metrics + other_metrics


@pytest.fixture()
def included_datasources():
    "Returns a thorough but not necessarily exhaustive list of data sources"
    desktop_datasources = [
        ConfigLoader.get_data_source("clients_daily", "firefox_desktop"),
        ConfigLoader.get_data_source(
            "search_clients_engines_sources_daily", "firefox_desktop"
        ),
        ConfigLoader.get_data_source("main_summary", "firefox_desktop"),
        ConfigLoader.get_data_source("events", "firefox_desktop"),
        ConfigLoader.get_data_source("normandy_events", "firefox_desktop"),
        ConfigLoader.get_data_source("main", "firefox_desktop"),
        ConfigLoader.get_data_source("crash", "firefox_desktop"),
        ConfigLoader.get_data_source("cfr", "firefox_desktop"),
        ConfigLoader.get_data_source("activity_stream_events", "firefox_desktop"),
    ]

    other_datasources = []
    for platform in [
        "fenix",
        "firefox_ios",
        "focus_android",
        "focus_ios",
        "klar_android",
        "klar_ios",
    ]:
        other_datasources += [
            ConfigLoader.get_data_source("baseline", platform),
            ConfigLoader.get_data_source("events", platform),
            ConfigLoader.get_data_source("metrics", platform),
        ]

    return desktop_datasources + other_datasources


def test_sql_not_detectably_malformed(included_metrics, included_datasources):
    for m in included_metrics:
        sql_lint(m.select_expr.format(experiment_slug="slug"))

    for ds in included_datasources:
        sql_lint(ds.from_expr_for(None))


def test_included_metrics_have_docs(included_metrics):
    for m in included_metrics:
        assert m.friendly_name, m.name
        assert m.description, m.name
