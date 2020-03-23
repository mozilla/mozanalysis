# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozanalysis.segments import Segment, SegmentDataSource


clients_last_seen_lag_1_day = SegmentDataSource(
    name='clients_last_seen_lag_1_day',
    from_expr="`moz-fx-data-shared-prod.telemetry.clients_last_seen`",
    window_start=-1,
    window_end=-1
)


# Before https://github.com/mozilla/bigquery-etl/pull/825/ is merged,
# we gotta do our own transforms
tmp_num_weeks_gte_2_5_uri = """COALESCE(
                CAST(
                    BIT_COUNT(
                        days_visited_5_uri_bits
                        & `moz-fx-data-shared-prod`.udf.bitmask_range(1, 7)
                    ) >= 2 AS INT64
                ) + CAST(
                    BIT_COUNT(
                        days_visited_5_uri_bits
                        & `moz-fx-data-shared-prod`.udf.bitmask_range(8, 7)
                    ) >= 2 AS INT64
                ) + CAST(
                    BIT_COUNT(
                        days_visited_5_uri_bits
                        & `moz-fx-data-shared-prod`.udf.bitmask_range(15, 7)
                    ) >= 2 AS INT64
                ) + CAST(
                    BIT_COUNT(
                        days_visited_5_uri_bits
                        & `moz-fx-data-shared-prod`.udf.bitmask_range(22, 7)
                    ) >= 2 AS INT64
                ),
                0
            )"""


regular_users_v1 = Segment(
    name='regular_users_v1',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="MAX(segment_regular_user_v1 = 'regular_users_v1')",
    select_expr=f"MAX({tmp_num_weeks_gte_2_5_uri} = 4)",
)

semi_regular_users_v1 = Segment(
    name='semi_regular_users_v1',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="MAX(segment_regular_user_v1 = 'semi_regular_users_v1')",
    select_expr=f"MAX({tmp_num_weeks_gte_2_5_uri} BETWEEN 1 AND 3)",
)

new_irregular_users_v1 = Segment(
    name='new_irregular_users_v1',
    data_source=clients_last_seen_lag_1_day,
    # select_expr="MAX(segment_regular_user_v1 = 'new_irregular_users_v1')",
    select_expr=f"MAX({tmp_num_weeks_gte_2_5_uri} = 0)",
)
