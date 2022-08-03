# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from textwrap import dedent

from mozanalysis.metrics import agg_any
from mozanalysis.historical_targets import TargetDataSource, Target

#: SegmentDataSource: The clients_last_seen table.
clients_last_seen = TargetDataSource(
    name="clients_last_seen",
    from_expr="mozdata.telemetry.clients_last_seen"
)

#: Segment: ...
regular_users_v3 = Target(
    name="regular_users_v3",
    data_source=clients_last_seen,
    where_expr="is_regular_user_v3",
    friendly_name="Regular users (v3)",
    description=dedent(
        """\
        Clients who used Firefox on at least 14 of the 27 days prior to enrolling.
        This segment is characterized by high retention.
        """
    ),
)

#: Segment: ...
new_or_resurrected_v3 = Target(
    name="new_or_resurrected_v3",
    data_source=clients_last_seen,
    where_expr="is_new_or_resurrected_v3",
    friendly_name="New or resurrected users (v3)",
    description=dedent(
        """\
        Clients who used Firefox on none of the 27 days prior to enrolling.
        """
    ),
)

#: Segment: ...
weekday_regular_v1 = Target(
    name="weekday_regular_v1",
    data_source=clients_last_seen,
    where_expr="is_weekday_regular_v1",
    friendly_name="Weekday regular users (v1)",
    description=dedent(
        """\
        A subset of "regular users" who typically use Firefox on weekdays.
        """
    ),
)

#: Segment: ...
allweek_regular_v1 = Target(
    name="allweek_regular_v1",
    data_source=clients_last_seen,
    where_expr="is_allweek_regular_v1",
    friendly_name="All-week regulars (v1)",
    description=dedent(
        """\
        A subset of "regular users" that have used Firefox on weekends.
        """
    ),
)

#: Segment: ...
new_unique_profiles = Target(
    name="new_unique_profiles",
    data_source=clients_last_seen,
    where_expr="first_seen_date >= submission_date",
    friendly_name="New unique profiles",
    description=dedent(
        """\
        Clients that enrolled the first date their client_id ever appeared
        in telemetry (i.e. new, unique profiles).
    """
    ),
)
