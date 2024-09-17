# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import attr
from metric_config_parser import AnalysisUnit
from typing_extensions import assert_never

from mozanalysis.metrics import DataSource


@attr.s(frozen=True, slots=True)
class ExposureSignal:
    """Represents a signal that determines an exposure has happened.

    The optional ``window_start`` and ``window_end`` parameters define
    the window of data used to determine whether each client has been exposed.
    ``window_start`` and ``window_end`` are integers, representing the number
    of days before or after the first enrollment date.
    If ``window_start`` is set to None, then by default the first enrollment date
    is used. If ``window_end`` is set to None, then by default the last enrollment
    date is used.


    Args:
        name (str): A slug; uniquely identifies this exposure signal in tables
        data_source (DataSource): data source to query from
        select_expr (str): a SQL snippet representing a clause of a SELECT
            expression describing how to compute the exposure signal; must
            be a boolean condition
        friendly_name (str): A human-readable dashboard title for this exposure signal
        description (str): A paragraph of Markdown-formatted text describing
            what the exposure signal represents, to be shown on dashboards
        window_start (int): Optional, see above
        window_end (int): Optional, see above
    """

    name = attr.ib(type=str)
    data_source = attr.ib(type=DataSource)
    select_expr = attr.ib(type=str)
    friendly_name = attr.ib(type=str | None, default=None)
    description = attr.ib(type=str | None, default=None)
    window_start = attr.ib(type=str | None, default=None)
    window_end = attr.ib(type=str | None, default=None)

    def build_query(
        self,
        time_limits,
        analysis_unit: AnalysisUnit = AnalysisUnit.CLIENT,
    ):
        """Return a nearly self-contained query for determining exposures.

        This query does not define ``enrollments`` but otherwise could
        be executed to query all exposures based on the exposure metric
        from this data source.
        """
        if analysis_unit == AnalysisUnit.CLIENT:
            ds_id = self.data_source.client_id_column
        elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
            ds_id = self.data_source.group_id_column
        else:
            assert_never(analysis_unit)
        return """SELECT
            e.{analysis_id},
            e.branch,
            MIN(ds.submission_date) AS exposure_date,
            COUNT(ds.submission_date) AS num_exposure_events
        FROM raw_enrollments e
            LEFT JOIN (
                SELECT
                    {ds_id} AS {analysis_id},
                    {submission_date} AS submission_date
                FROM {from_expr}
                WHERE {submission_date}
                    BETWEEN DATE_ADD('{date_start}', INTERVAL {window_start} DAY)
                    AND DATE_ADD('{date_end}', INTERVAL {window_end} DAY)
                    AND {exposure_signal}
            ) AS ds
            ON ds.{analysis_id} = e.{analysis_id} AND
                ds.submission_date >= e.enrollment_date
        GROUP BY
            e.{analysis_id},
            e.branch""".format(
            ds_id=ds_id,
            submission_date=self.data_source.submission_date_column,
            from_expr=self.data_source.from_expr_for(None),
            date_start=time_limits.first_enrollment_date,
            date_end=time_limits.first_enrollment_date
            if self.window_end
            else time_limits.last_enrollment_date,
            window_start=self.window_start or 0,
            window_end=self.window_end or 0,
            exposure_signal=self.select_expr,
            analysis_id=analysis_unit.value,
        )
