# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import attr
from metric_config_parser import AnalysisUnit
from typing_extensions import assert_never

from mozanalysis.types import IncompatibleAnalysisUnit


@attr.s(frozen=True, slots=True)
class SegmentDataSource:
    """Represents a table or view, from which segments may be defined.

    ``window_start`` and ``window_end`` define the window of data used
    to determine whether each client fits a segment. Ideally this
    window ends at/before the moment of enrollment, so that user's
    branches can't bias the segment assignment. ``window_start`` and
    ``window_end`` are integers, representing the number
    of days before or after enrollment.


    Args:
        name (str): Name for the Data Source. Should be unique to avoid
            confusion.
        from_expr (str): FROM expression - often just a fully-qualified
            table name. Sometimes a subquery. May contain the string
            ``{dataset}`` which will be replaced with an app-specific
            dataset for Glean apps. If the expression is templated
            on dataset, default_dataset is mandatory.
        window_start (int, optional): See above.
        window_end (int, optional): See above.
        client_id_column (str, optional): Name of the column that
            contains the ``client_id`` (join key). Defaults to
            'client_id'.
        submission_date_column (str, optional): Name of the column
            that contains the submission date (as a date, not
            timestamp). Defaults to 'submission_date'.
        default_dataset (str, optional): The value to use for
            `{dataset}` in from_expr if a value is not provided
            at runtime. Mandatory if from_expr contains a
            `{dataset}` parameter.
        app_name: (str, optional): app_name used with metric-hub,
            used for validation
        group_id_column (str, optional): Name of the column that
            contains the ``group_id`` (join key). Defaults to
            'profile_group_id'.
    """

    name = attr.ib(validator=attr.validators.instance_of(str))
    _from_expr = attr.ib(validator=attr.validators.instance_of(str))
    window_start = attr.ib(default=0, type=int)
    window_end = attr.ib(default=0, type=int)
    client_id_column = attr.ib(default=AnalysisUnit.CLIENT.value, type=str)
    submission_date_column = attr.ib(default="submission_date", type=str)
    default_dataset = attr.ib(default=None, type=str | None)
    app_name = attr.ib(default=None, type=str | None)
    group_id_column = attr.ib(default=AnalysisUnit.PROFILE_GROUP.value, type=str)

    @default_dataset.validator
    def _check_default_dataset_provided_if_needed(self, attribute, value):
        self.from_expr_for(None)

    def from_expr_for(self, dataset: str | None) -> str:
        """Expands the ``from_expr`` template for the given dataset.
        If ``from_expr`` is not a template, returns ``from_expr``.

        Args:
            dataset (str or None): Dataset name to substitute
                into the from expression.
        """
        effective_dataset = dataset or self.default_dataset
        if effective_dataset is None:
            try:
                return self._from_expr.format()
            except Exception as e:
                raise ValueError(
                    f"{self.name}: from_expr contains a dataset template but no value was provided."  # noqa:E501
                ) from e
        return self._from_expr.format(dataset=effective_dataset)

    def build_query(
        self,
        segment_list,
        time_limits,
        experiment_slug,
        from_expr_dataset=None,
        analysis_unit: AnalysisUnit = AnalysisUnit.CLIENT,
    ):
        """Return a nearly self contained SQL query.

        The query takes a list of ``{analysis_id}``s from
        ``raw_enrollments``, and adds one non-NULL boolean column per
        segment: True if the client is in the segment, False otherwise.
        """
        if analysis_unit == AnalysisUnit.CLIENT:
            ds_id = self.client_id_column
        elif analysis_unit == AnalysisUnit.PROFILE_GROUP:
            ds_id = self.group_id_column
        else:
            assert_never(analysis_unit)
        return """SELECT
            e.{analysis_id},
            e.branch,
            {segments}
        FROM raw_enrollments e
            LEFT JOIN {from_expr} ds
                ON ds.{ds_id} = e.{analysis_id}
                AND ds.{submission_date} BETWEEN
                    DATE_ADD('{first_enrollment}', interval {window_start} day)
                    AND DATE_ADD('{last_enrollment}', interval {window_end} day)
                AND ds.{submission_date} BETWEEN
                    DATE_ADD(e.enrollment_date, interval {window_start} day)
                    AND DATE_ADD(e.enrollment_date, interval {window_end} day)
        GROUP BY e.{analysis_id}, e.branch""".format(
            ds_id=ds_id,
            submission_date=self.submission_date_column or "submission_date",
            from_expr=self.from_expr_for(from_expr_dataset),
            first_enrollment=time_limits.first_enrollment_date,
            last_enrollment=time_limits.last_enrollment_date,
            window_start=self.window_start,
            window_end=self.window_end,
            segments=",\n            ".join(
                f"{m.select_expr} AS {m.name}" for m in segment_list
            ),
            analysis_id=analysis_unit.value,
        )

    def build_query_target(
        self,
        target,
        time_limits,
        from_expr_dataset=None,
        analysis_unit: AnalysisUnit = AnalysisUnit.CLIENT,
    ):
        """
        Return a nearly-self contained SQL query, for use with
        mozanalysis.sizing.HistoricalTarget.

        This query returns all distinct client IDs that satisfy the criteria
        for inclusion in a historical analysis using this datasource.
        Separate sub-queries are constructed for each additional Segment
        in the analysis.
        """
        if analysis_unit != AnalysisUnit.CLIENT:
            raise IncompatibleAnalysisUnit(
                "`build_query_targets` currently only supports client_id analysis"
            )

        return """
        SELECT
            {client_id} as client_id,
            target_first_date,
            target_last_date,
            {target_name}
        FROM (SELECT {client_id},
                MIN({submission_date}) as target_first_date,
                MAX({submission_date}) as target_last_date,
                {target}
            FROM {from_expr}
            WHERE {submission_date} BETWEEN '{fddr}' AND '{lddr}'
            GROUP BY {client_id})
        WHERE {target_name}""".format(
            client_id=self.client_id_column or "client_id",
            submission_date=self.submission_date_column or "submission_date",
            from_expr=self.from_expr_for(from_expr_dataset),
            fddr=time_limits.first_enrollment_date,
            lddr=time_limits.last_enrollment_date,
            target=f"{target.select_expr} AS {target.name}",
            target_name=target.name,
        )

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
            an aggregation function (we ``GROUP BY {analysis_unit}``).
            Returns a non-NULL ``BOOL``: ``True`` if the user is in the
            segment, ``False`` otherwise.
        friendly_name (str): A human-readable dashboard title for this segment
        description (str): A paragraph of Markdown-formatted text describing
            the segment in more detail, to be shown on dashboards
        app_name: (str, optional): app_name used with metric-hub,
            used for validation
    """

    name = attr.ib(type=str)
    data_source = attr.ib(validator=attr.validators.instance_of(SegmentDataSource))
    select_expr = attr.ib(type=str)
    friendly_name = attr.ib(type=str | None, default=None)
    description = attr.ib(type=str | None, default=None)
    app_name = attr.ib(type=str | None, default=None)
