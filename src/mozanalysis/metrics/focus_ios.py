# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.config import ConfigLoader

baseline = ConfigLoader.get_data_source("baseline", "focus_ios")

events = ConfigLoader.get_data_source("events", "focus_ios")

metrics = ConfigLoader.get_data_source("metrics", "focus_ios")

baseline_ping_count = ConfigLoader.get_metric("baseline_ping_count", "focus_ios")

metric_ping_count = ConfigLoader.get_metric("metric_ping_count", "focus_ios")

first_run_date = ConfigLoader.get_metric("first_run_date", "focus_ios")
