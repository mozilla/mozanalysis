# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# These metric definitions are deprecated.
# Instead use the metric slug to reference metrics defined
# in https://github.com/mozilla/metric-hub

from mozanalysis.config import ConfigLoader

baseline = ConfigLoader.get_data_source("baseline", "fenix")

events = ConfigLoader.get_data_source("events", "fenix")

metrics = ConfigLoader.get_data_source("metrics", "fenix")


uri_count = ConfigLoader.get_metric("uri_count", "fenix")

user_reports_site_issue_count = ConfigLoader.get_metric(
    "user_reports_site_issue_count", "fenix"
)

user_reload_count = ConfigLoader.get_metric("user_reload_count", "fenix")

baseline_ping_count = ConfigLoader.get_metric("baseline_ping_count", "fenix")

metric_ping_count = ConfigLoader.get_metric("metric_ping_count", "fenix")

first_run_date = ConfigLoader.get_metric("first_run_date", "fenix")
