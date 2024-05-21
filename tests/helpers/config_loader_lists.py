from mozanalysis.config import ConfigLoader

desktop_segments = [
    ConfigLoader.get_segment("regular_users_v3", "firefox_desktop"),
    ConfigLoader.get_segment("new_or_resurrected_v3", "firefox_desktop"),
    ConfigLoader.get_segment("weekday_regular_v1", "firefox_desktop"),
    ConfigLoader.get_segment("allweek_regular_v1", "firefox_desktop"),
    ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop"),
]

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
