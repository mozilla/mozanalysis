from mozanalysis.config import ConfigLoader

desktop_segments = [
    ConfigLoader.get_segment("regular_users_v3", "firefox_desktop"),
    ConfigLoader.get_segment("new_or_resurrected_v3", "firefox_desktop"),
    ConfigLoader.get_segment("weekday_regular_v1", "firefox_desktop"),
    ConfigLoader.get_segment("allweek_regular_v1", "firefox_desktop"),
    ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop"),
]
