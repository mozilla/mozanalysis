# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# These segment definitions are deprecated.
# Instead use the segment slug to reference segments defined
# in https://github.com/mozilla/metric-hub

from mozanalysis.config import ConfigLoader

clients_last_seen = ConfigLoader.get_segment_data_source(
    "clients_last_seen", "firefox_desktop"
)


regular_users_v3 = ConfigLoader.get_segment("regular_users_v3", "firefox_desktop")

new_or_resurrected_v3 = ConfigLoader.get_segment(
    "new_or_resurrected_v3", "firefox_desktop"
)

weekday_regular_v1 = ConfigLoader.get_segment("weekday_regular_v1", "firefox_desktop")

allweek_regular_v1 = ConfigLoader.get_segment("allweek_regular_v1", "firefox_desktop")

new_unique_profiles = ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop")
