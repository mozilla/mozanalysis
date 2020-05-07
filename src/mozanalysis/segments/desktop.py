# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.segments import Segment, SegmentDataSource


clients_last_seen_lag_1_day = SegmentDataSource(
    name='clients_last_seen',
    from_expr="`moz-fx-data-shared-prod.telemetry.clients_last_seen`",
    window_start=0,
    window_end=0,
)


regular_users_v3 = Segment(
    name='regular_users_v3',
    data_source=clients_last_seen_lag_1_day,
    select_expr=f"""MAX(
        BIT_COUNT(COALESCE(days_seen_bits, 0) & 0x0FFFFFFE) >= 14
    )""",
)

new_or_resurrected_v3 = Segment(
    name='new_or_resurrected_v3',
    data_source=clients_last_seen_lag_1_day,
    select_expr=f"""MAX(
        BIT_COUNT(COALESCE(days_seen_bits, 0) & 0x0FFFFFFE) = 0
    )""",
)
