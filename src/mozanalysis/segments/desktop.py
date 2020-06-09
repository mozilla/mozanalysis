# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.metrics import agg_any
from mozanalysis.segments import Segment, SegmentDataSource


clients_last_seen = SegmentDataSource(
    name='clients_last_seen',
    from_expr="`moz-fx-data-shared-prod.telemetry.clients_last_seen`",
    window_start=0,
    window_end=0,
)


regular_users_v3 = Segment(
    name='regular_users_v3',
    data_source=clients_last_seen,
    select_expr=agg_any('is_regular_user_v3'),
)

new_or_resurrected_v3 = Segment(
    name='new_or_resurrected_v3',
    data_source=clients_last_seen,
    select_expr="LOGICAL_OR(COALESCE(is_new_or_resurrected_v3, TRUE))",
)


weekday_regular_v1 = Segment(
    name='weekday_regular_v1',
    data_source=clients_last_seen,
    select_expr=agg_any('is_weekday_regular_v1'),
)

allweek_regular_v1 = Segment(
    name='allweek_regular_v1',
    data_source=clients_last_seen,
    select_expr=agg_any('is_allweek_regular_v1'),
)
