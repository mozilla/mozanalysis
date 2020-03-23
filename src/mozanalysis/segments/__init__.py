# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr


@attr.s(frozen=True, slots=True)
class SegmentDataSource:
    """Represents a table or view, from which segments may be defined.

    ``window_start`` and ``window_end`` define the window of data used
    to determine whether each client fits a segment. Ideally this
    window ends at/before the moment of enrollment, so that user's
    branches can't bias the segment assignment. ``window_start`` and
    ``window_end`` are non-positive integers, representing the number
    of days before enrollment::

        window_start <= window_end <= 0


    Args:
        name (str): Name for the Data Source. Should be unique to avoid
            confusion.
        from_expr (str): FROM expression - often just a fully-qualified
            table name. Sometimes a subquery.
        window_start (int, optional): See above.
        window_end (int, optional): See above.
        client_id_column (str, optional): Name of the column that
            contains the ``client_id`` (join key). Defaults to
            'client_id'.
        submission_date_column (str, optional): Name of the column
            that contains the submission date (as a date, not
            timestamp). Defaults to 'submission_date'.
    """
    name = attr.ib(validator=attr.validators.instance_of(str))
    from_expr = attr.ib(validator=attr.validators.instance_of(str))
    window_start = attr.ib(default=0, type=int)
    window_end = attr.ib(default=0, type=int)
    client_id_column = attr.ib(default='client_id', type=str)
    submission_date_column = attr.ib(default='submission_date', type=str)

    def build_query(self, segment_list, time_limits, experiment_slug):
        """Return a nearly self contained SQL query.

        The query takes a list of ``client_id``s from
        ``raw_enrollments``, and adds one non-NULL boolean column per
        segment: True if the client is in the segment, False otherwise.
        """
        return """SELECT
            e.client_id,
            {segments}
        FROM raw_enrollments e
            LEFT JOIN {from_expr} ds
                ON ds.{client_id} = e.client_id
                AND ds.{submission_date} BETWEEN
                    DATE_ADD('{first_enrollment}', interval {window_start} day)
                    AND DATE_ADD('{last_enrollment}', interval {window_end} day)
                AND ds.{submission_date} BETWEEN
                    DATE_ADD(e.enrollment_date, interval {window_start} day)
                    AND DATE_ADD(e.enrollment_date, interval {window_end} day)
        GROUP BY e.client_id""".format(
            client_id=self.client_id_column,
            submission_date=self.submission_date_column,
            from_expr=self.from_expr,
            first_enrollment=time_limits.first_enrollment_date,
            last_enrollment=time_limits.last_enrollment_date,
            window_start=self.window_start,
            window_end=self.window_end,
            segments=',\n            '.join(
                f"{m.select_expr} AS {m.name}" for m in segment_list
            ),
        )

    @window_end.validator
    def window_end_is_not_positive(self, attribute, value):
        if value > 0:
            raise ValueError("window_end must be <= 0")

    @window_start.validator
    def window_start_lte_window_end(self, attribute, value):
        if value > self.window_end:
            raise ValueError("window_start must be <= window_end")


@attr.s(frozen=True, slots=True)
class Segment:
    """Represents an experiment Segment.

    Args:
        name (str): The segment's name; will be a column name.
        data_source (SegmentDataSource): Data source that provides
            the columns referenced in ``select_expr``.
        select_expr (str): A SQL select expression that includes
            an aggregation function (we ``GROUP BY client_id``).
            Returns a non-NULL ``BOOL``: ``True`` if the user is in the
            segment, ``False`` otherwise.
    """
    name = attr.ib(type=str)
    data_source = attr.ib(validator=attr.validators.instance_of(SegmentDataSource))
    select_expr = attr.ib(type=str)
