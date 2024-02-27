# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# These metric definitions are deprecated.
# Instead use the metric slug to reference metrics defined
# in https://github.com/mozilla/metric-hub

from mozanalysis.config import ConfigLoader
import warnings

warnings.simplefilter("default")
warnings.warn(
    """
    metrics and data sources created in mozanalysis are deprecated
    please create directly from metric-hub with ConfigLoader like

    from mozanalysis.config import ConfigLoader
    metric=ConfigLoader.get_metric(metric_slug="active_hours",app_name="firefox_desktop")

    and data sources like
    data_source=ConfigLoader.get_data_source(data_source_slug="active_hours",
                                                app_name="firefox_desktop")

    """,
    DeprecationWarning,
)

baseline = ConfigLoader.get_data_source("baseline", "klar_android")

events = ConfigLoader.get_data_source("events", "klar_android")

metrics = ConfigLoader.get_data_source("metrics", "klar_android")


baseline_ping_count = ConfigLoader.get_metric("baseline_ping_count", "klar_android")

metric_ping_count = ConfigLoader.get_metric("metric_ping_count", "klar_android")

first_run_date = ConfigLoader.get_metric("first_run_date", "klar_android")
