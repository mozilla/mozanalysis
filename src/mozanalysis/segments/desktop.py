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


# Before https://github.com/mozilla/bigquery-etl/pull/825/ is merged,
# we gotta do our own transforms
tmp_segment_regular_users_v2 = """  CASE
  WHEN
    BIT_COUNT(days_visited_5_uri_bits & 0x0FFFFFFE) = 0
  THEN
    'new_irregular_users_v2'
  WHEN
    BIT_COUNT(days_visited_5_uri_bits & 0x0FFFFFFE)
    BETWEEN 1
    AND 7
  THEN
    'semi_regular_users_v2'
  WHEN
    BIT_COUNT(days_visited_5_uri_bits & 0x0FFFFFFE)
    BETWEEN 8
    AND 27
  THEN
    'regular_users_v2'
  END
"""


regular_users_v2 = Segment(
    name='regular_users_v2',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="""MAX(
    #     COALESCE(segment_regular_users_v2, 'new_irregular_users_v2')
    #     = 'regular_users_v2'
    # )""",
    select_expr=f"""MAX(
        COALESCE({tmp_segment_regular_users_v2}, 'new_irregular_users_v2')
        = 'regular_users_v2'
    )""",
)

semi_regular_users_v2 = Segment(
    name='semi_regular_users_v2',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="""MAX(
    #     COALESCE(segment_regular_users_v2, 'new_irregular_users_v2')
    #     = 'semi_regular_users_v2'
    # )""",
    select_expr=f"""MAX(
        COALESCE({tmp_segment_regular_users_v2}, 'new_irregular_users_v2')
        = 'semi_regular_users_v2'
    )""",
)

new_irregular_users_v2 = Segment(
    name='new_irregular_users_v2',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="""MAX(
    #     COALESCE(segment_regular_users_v2, 'new_irregular_users_v2')
    #     = 'new_irregular_users_v2'
    # )""",
    select_expr=f"""MAX(
        COALESCE({tmp_segment_regular_users_v2}, 'new_irregular_users_v2')
        = 'new_irregular_users_v2'
    )""",
)
