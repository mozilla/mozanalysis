# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional

import attr

from mozanalysis.metrics import DataSource


@attr.s(frozen=True, slots=True)
class ExposureSignal:
    """Represents a signal that determines an exposure has happened.

    Args:
        name (str): A slug; uniquely identifies this exposure signal in tables
        data_source (DataSource): data source to query from
        select_expr (str): a SQL snippet representing a clause of a SELECT
            expression describing how to compute the exposure signal; must
            be a boolean condition
        friendly_name (str): A human-readable dashboard title for this exposure signal
        description (str): A paragraph of Markdown-formatted text describing
            what the exposure signal represents, to be shown on dashboards
    """

    name = attr.ib(type=str)
    data_source = attr.ib(type=DataSource)
    select_expr = attr.ib(type=str)
    friendly_name = attr.ib(type=Optional[str], default=None)
    description = attr.ib(type=Optional[str], default=None)

    def build_query(
        self,
        time_limits,
    ):
        """Return a nearly self-contained query for determining exposures.

        This query does not define ``enrollments`` but otherwise could
        be executed to query all exposures based on the exposure metric
        from this data source.
        """
        return """SELECT
            e.client_id,
            e.branch,
            MIN(ds.submission_date) AS exposure_date,
            COUNT(ds.submission_date) AS num_exposure_events
        FROM raw_enrollments e
            LEFT JOIN (
                SELECT
                    {client_id} AS client_id,
                    {submission_date} AS submission_date
                FROM {from_expr}
                WHERE {submission_date}
                    BETWEEN '{first_enrollment_date}' AND '{last_enrollment_date}'
                    AND {exposure_signal}
            ) AS ds
            ON ds.client_id = e.client_id AND
                ds.submission_date >= e.enrollment_date
        GROUP BY
            e.client_id,
            e.branch""".format(
            client_id=self.data_source.client_id_column,
            submission_date=self.data_source.submission_date_column,
            from_expr=self.data_source.from_expr_for(None),
            first_enrollment_date=time_limits.first_enrollment_date,
            last_enrollment_date=time_limits.last_enrollment_date,
            exposure_signal=self.select_expr,
        )
