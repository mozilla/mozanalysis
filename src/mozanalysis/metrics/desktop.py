# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# These metric definitions are deprecated.
# Instead use the metric slug to reference metrics defined
# in https://github.com/mozilla/metric-hub

from mozanalysis.config import ConfigLoader

clients_daily = ConfigLoader.get_data_source("clients_daily", "firefox_desktop")

search_clients_engines_sources_daily = ConfigLoader.get_data_source(
    "search_clients_engines_sources_daily", "firefox_desktop"
)

search_clients_daily = search_clients_engines_sources_daily

main_summary = ConfigLoader.get_data_source("main_summary", "firefox_desktop")

events = ConfigLoader.get_data_source("events", "firefox_desktop")

normandy_events = ConfigLoader.get_data_source("normandy_events", "firefox_desktop")

main = ConfigLoader.get_data_source("main", "firefox_desktop")

crash = ConfigLoader.get_data_source("crash", "firefox_desktop")

cfr = ConfigLoader.get_data_source("cfr", "firefox_desktop")

activity_stream_events = ConfigLoader.get_data_source(
    "activity_stream_events", "firefox_desktop"
)


active_hours = ConfigLoader.get_metric("active_hours", "firefox_desktop")

uri_count = ConfigLoader.get_metric("uri_count", "firefox_desktop")

search_count = ConfigLoader.get_metric("search_count", "firefox_desktop")

tagged_search_count = ConfigLoader.get_metric("tagged_search_count", "firefox_desktop")

tagged_follow_on_search_count = ConfigLoader.get_metric(
    "tagged_follow_on_search_count", "firefox_desktop"
)

ad_clicks = ConfigLoader.get_metric("ad_clicks", "firefox_desktop")

searches_with_ads = ConfigLoader.get_metric("searches_with_ads", "firefox_desktop")

organic_search_count = ConfigLoader.get_metric(
    "organic_search_count", "firefox_desktop"
)

unenroll = ConfigLoader.get_metric("unenroll", "firefox_desktop")

view_about_logins = ConfigLoader.get_metric("view_about_logins", "firefox_desktop")

view_about_protections = ConfigLoader.get_metric(
    "view_about_protections", "firefox_desktop"
)

connect_fxa = ConfigLoader.get_metric("connect_fxa", "firefox_desktop")

pocket_rec_clicks = ConfigLoader.get_metric("pocket_rec_clicks", "firefox_desktop")

pocket_spoc_clicks = ConfigLoader.get_metric("pocket_spoc_clicks", "firefox_desktop")

days_of_use = ConfigLoader.get_metric("days_of_use", "firefox_desktop")

qualified_cumulative_days_of_use = ConfigLoader.get_metric(
    "qualified_cumulative_days_of_use", "firefox_desktop"
)

disable_pocket_clicks = ConfigLoader.get_metric(
    "disable_pocket_clicks", "firefox_desktop"
)

disable_pocket_spocs_clicks = ConfigLoader.get_metric(
    "disable_pocket_spocs_clicks", "firefox_desktop"
)
