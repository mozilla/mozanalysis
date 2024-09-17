# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from metric_config_parser import AnalysisUnit
from typing_extensions import assert_never

from mozanalysis.types import IncompatibleAnalysisUnit

if TYPE_CHECKING:
    from metric_config_parser.data_source import DataSource as ParserDataSource

    from mozanalysis.experiment import TimeLimits

import logging

import attr

logger = logging.getLogger(__name__)


class AnalysisBasis(Enum):
    """Determines what the population used for the analysis will be based on."""

    ENROLLMENTS = "enrollments"
    EXPOSURES = "exposures"


# attr.s converters aren't compatible with mypy, define our own
# see: https://mypy.readthedocs.io/en/stable/additional_features.html#id1
def client_id_column_converter(client_id_column: str | None) -> str:
    if client_id_column is None:
        return AnalysisUnit.CLIENT.value
    else:
        return client_id_column


def group_id_column_converter(group_id_column: str | None) -> str:
    if group_id_column is None:
        return AnalysisUnit.PROFILE_GROUP.value
    else:
        return group_id_column


def submission_date_column_converter(submission_date_column: str | None) -> str:
    if submission_date_column is None:
        return "submission_date"
    else:
        return submission_date_column


@attr.s(frozen=True, slots=True)
class DataSource:
    """Represents a table or view, from which Metrics may be defined.

    Args:
        name (str): Name for the Data Source. Used in sanity metric
            column names.
        from_expr (str): FROM expression - often just a fully-qualified
            table name. Sometimes a subquery. May contain the string
            ``{dataset}`` which will be replaced with an app-specific
            dataset for Glean apps. If the expression is templated
            on dataset, default_dataset is mandatory.
        experiments_column_type (str or None): Info about the schema
            of the table or view:

            * 'simple': There is an ``experiments`` column, which is an
              (experiment_slug:str -> branch_name:str) map.
            * 'native': There is an ``experiments`` column, which is an
              (experiment_slug:str -> struct) map, where the struct
              contains a ``branch`` field, which is the branch as a
              string.
            * None: There is no ``experiments`` column, so skip the
              sanity checks that rely on it. We'll also be unable to
              filter out pre-enrollment data from day 0 in the
              experiment.
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
    """

    name = attr.ib(validator=attr.validators.instance_of(str))
    _from_expr = attr.ib(validator=attr.validators.instance_of(str))
    experiments_column_type = attr.ib(default="simple", type=str | None)
    client_id_column = attr.ib(
        default=AnalysisUnit.CLIENT.value,
        type=str,
        validator=[attr.validators.instance_of(str), attr.validators.min_len(1)],
        converter=client_id_column_converter,
    )
    submission_date_column = attr.ib(
        default="submission_date",
        type=str,
        validator=[attr.validators.instance_of(str), attr.validators.min_len(1)],
        converter=submission_date_column_converter,
    )
    default_dataset = attr.ib(default=None, type=str | None)
    app_name = attr.ib(default=None, type=str | None)
    group_id_column = attr.ib(
        default=AnalysisUnit.PROFILE_GROUP.value,
        type=str,
        validator=[attr.validators.instance_of(str), attr.validators.min_len(1)],
        converter=group_id_column_converter,
    )

    EXPERIMENT_COLUMN_TYPES = (None, "simple", "native", "glean")

    @experiments_column_type.validator
    def _check_experiments_column_type(self, attribute, value):
        if value not in self.EXPERIMENT_COLUMN_TYPES:
            raise ValueError(
                f"experiments_column_type {value!r} must be one of: "
                f"{self.EXPERIMENT_COLUMN_TYPES!r}"
            )

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

    @property
    def experiments_column_expr(self) -> str:
        if self.experiments_column_type is None:
            return ""

        elif self.experiments_column_type == "simple":
            return """AND (
                    ds.{submission_date} != e.enrollment_date
                    OR `mozfun.map.get_key`(
                        ds.experiments, '{experiment_slug}'
                    ) IS NOT NULL
                )"""

        elif self.experiments_column_type == "native":
            return """AND (
                    ds.{submission_date} != e.enrollment_date
                    OR `mozfun.map.get_key`(
                        ds.experiments, '{experiment_slug}'
                    ).branch IS NOT NULL
            )"""

        elif self.experiments_column_type == "glean":
            return """AND (
                    ds.{submission_date} != e.enrollment_date
                    OR `mozfun.map.get_key`(
                        ds.ping_info.experiments, '{experiment_slug}'
                    ).branch IS NOT NULL
                )"""

        else:
            raise ValueError

    def build_query(
        self,
        metric_list: list[Metric],
        time_limits: TimeLimits,
        experiment_slug: str,
        from_expr_dataset: str | None = None,
        analysis_basis: AnalysisBasis = AnalysisBasis.ENROLLMENTS,
        analysis_unit: AnalysisUnit = AnalysisUnit.CLIENT,
        exposure_signal=None,
    ) -> str:
        """Return a nearly-self contained SQL query.

        This query does not define ``enrollments`` but otherwise could
        be executed to query all metrics from this data source.
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
            e.analysis_window_start,
            e.analysis_window_end,
            e.num_exposure_events,
            e.exposure_date,
            {metrics}
        FROM enrollments e
            LEFT JOIN {from_expr} ds
                ON ds.{ds_id} = e.{analysis_id}
                AND ds.{submission_date} BETWEEN '{fddr}' AND '{lddr}'
                AND ds.{submission_date} BETWEEN
                    DATE_ADD(e.{date}, interval e.analysis_window_start day)
                    AND DATE_ADD(e.{date}, interval e.analysis_window_end day)
                {ignore_pre_enroll_first_day}
        GROUP BY
            e.{analysis_id},
            e.branch,
            e.num_exposure_events,
            e.exposure_date,
            e.analysis_window_start,
            e.analysis_window_end""".format(
            ds_id=ds_id,
            submission_date=self.submission_date_column,
            from_expr=self.from_expr_for(from_expr_dataset),
            fddr=time_limits.first_date_data_required,
            lddr=time_limits.last_date_data_required,
            metrics=",\n            ".join(
                f"{m.select_expr.format(experiment_slug=experiment_slug)} AS {m.name}"
                for m in metric_list
            ),
            date=(
                "exposure_date"
                if analysis_basis == AnalysisBasis.EXPOSURES and exposure_signal is None
                else "enrollment_date"
            ),
            ignore_pre_enroll_first_day=self.experiments_column_expr.format(
                submission_date=self.submission_date_column,
                experiment_slug=experiment_slug,
            ),
            analysis_id=analysis_unit.value,
        )

    def build_query_targets(
        self,
        metric_list: list[Metric],
        time_limits: TimeLimits,
        experiment_name: str,
        analysis_length: int,
        from_expr_dataset: str | None = None,
        continuous_enrollment: bool = False,
        analysis_unit: AnalysisUnit = AnalysisUnit.CLIENT,
    ) -> str:
        """Return a nearly-self contained SQL query that constructs
        the metrics query for targeting historical data without
        an associated experiment slug.

        This query does not define ``targets`` but otherwise could
        be executed to query all metrics from this data source.
        """
        if analysis_unit != AnalysisUnit.CLIENT:
            raise IncompatibleAnalysisUnit(
                "`build_query_targets` currently only supports client_id analysis"
            )

        return """
        SELECT
            t.client_id,
            t.enrollment_date,
            t.analysis_window_start,
            t.analysis_window_end,
            {metrics}
        FROM targets t
            LEFT JOIN {from_expr} ds
                ON ds.{client_id} = t.client_id
                {date_clause}
        GROUP BY
            t.client_id,
            t.enrollment_date,
            t.analysis_window_start,
            t.analysis_window_end""".format(
            client_id=self.client_id_column,
            from_expr=self.from_expr_for(from_expr_dataset),
            metrics=",\n            ".join(
                f"{m.select_expr.format(experiment_name=experiment_name)} AS {m.name}"
                for m in metric_list
            ),
            date_clause=(
                f"""
        AND ds.{self.submission_date_column} BETWEEN '{time_limits.first_date_data_required}' AND '{time_limits.last_date_data_required}'
        AND ds.{self.submission_date_column} BETWEEN
            DATE_ADD(t.enrollment_date, interval t.analysis_window_start day) AND
            DATE_ADD(t.enrollment_date, interval t.analysis_window_end day)"""  # noqa: E501
                if not continuous_enrollment
                else f"""AND ds.{self.submission_date_column} BETWEEN
            t.enrollment_date AND
            DATE_ADD(t.enrollment_date, interval {analysis_length} day)
            """
            ),
        )

    def get_sanity_metrics(self, experiment_slug: str) -> list[Metric]:
        if self.experiments_column_type is None:
            return []

        elif self.experiments_column_type == "simple":
            return [
                Metric(
                    name=self.name + "_has_contradictory_branch",
                    data_source=self,
                    select_expr=agg_any(
                        """`mozfun.map.get_key`(
                ds.experiments, '{experiment_slug}'
            ) != e.branch"""
                    ),
                ),
                Metric(
                    name=self.name + "_has_non_enrolled_data",
                    data_source=self,
                    select_expr=agg_any(
                        f"""`mozfun.map.get_key`(
                ds.experiments, '{experiment_slug}'
            ) IS NULL"""
                    ),
                ),
            ]

        elif self.experiments_column_type == "native":
            return [
                Metric(
                    name=self.name + "_has_contradictory_branch",
                    data_source=self,
                    select_expr=agg_any(
                        """`mozfun.map.get_key`(
                ds.experiments, '{experiment_slug}'
            ).branch != e.branch"""
                    ),
                ),
                Metric(
                    name=self.name + "_has_non_enrolled_data",
                    data_source=self,
                    select_expr=agg_any(
                        f"""`mozfun.map.get_key`(
                ds.experiments, '{experiment_slug}'
            ).branch IS NULL"""
                    ),
                ),
            ]

        elif self.experiments_column_type == "glean":
            return [
                Metric(
                    name=self.name + "_has_contradictory_branch",
                    data_source=self,
                    select_expr=agg_any(
                        """`mozfun.map.get_key`(
                ds.ping_info.experiments, '{experiment_slug}'
            ).branch != e.branch"""
                    ),
                ),
                Metric(
                    name=self.name + "_has_non_enrolled_data",
                    data_source=self,
                    select_expr=agg_any(
                        f"""`mozfun.map.get_key`(
                ds.ping_info.experiments, '{experiment_slug}'
            ).branch IS NULL"""
                    ),
                ),
            ]

        else:
            raise ValueError

    @classmethod
    def from_mcp_data_source(
        cls,
        parser_data_source: ParserDataSource,
        app_name: str | None = None,
        group_id_column: str | None = AnalysisUnit.PROFILE_GROUP.value,
    ) -> DataSource:
        """metric-config-parser DataSource objects do not have an `app_name`
        and do not, yet, have a group_id_column"""
        return cls(
            name=parser_data_source.name,
            from_expr=parser_data_source.from_expression,
            client_id_column=parser_data_source.client_id_column,
            submission_date_column=parser_data_source.submission_date_column,
            experiments_column_type=(
                None
                if parser_data_source.experiments_column_type == "none"
                else parser_data_source.experiments_column_type
            ),
            default_dataset=parser_data_source.default_dataset,
            app_name=app_name,
            group_id_column=group_id_column,
        )


@attr.s(frozen=True, slots=True)
class Metric:
    """Represents an experiment metric.

    Needs to be combined with an analysis window to be measurable!

    Args:
        name (str): A slug; uniquely identifies this metric in tables
        data_source (DataSource): where to find the metric
        select_expr (str): a SQL snippet representing a clause of a SELECT
            expression describing how to compute the metric; must include an
            aggregation function since it will be GROUPed BY the analysis unit
            and branch
        friendly_name (str): A human-readable dashboard title for this metric
        description (str): A paragraph of Markdown-formatted text describing
            what the metric measures, to be shown on dashboards
        app_name: (str, optional): app_name used with metric-hub,
            used for validation
    """

    name = attr.ib(type=str, validator=attr.validators.instance_of(str))
    data_source = attr.ib(
        type=DataSource, validator=attr.validators.instance_of(DataSource)
    )
    select_expr = attr.ib(type=str, validator=attr.validators.instance_of(str))
    friendly_name = attr.ib(type=str | None, default=None)
    description = attr.ib(type=str | None, default=None)
    bigger_is_better = attr.ib(type=bool, default=True)
    app_name = attr.ib(type=str | None, default=None)


def agg_sum(select_expr: str) -> str:
    """Return a SQL fragment for the sum over the data, with 0-filled nulls."""
    return f"COALESCE(SUM({select_expr}), 0)"


def agg_any(select_expr: str) -> str:
    """Return the logical OR, with FALSE-filled nulls."""
    return f"COALESCE(LOGICAL_OR({select_expr}), FALSE)"


def agg_histogram_mean(select_expr: str) -> str:
    """Produces an expression for the mean of an unparsed histogram."""
    return f"""SAFE_DIVIDE(
                SUM(CAST(JSON_EXTRACT_SCALAR({select_expr}, "$.sum") AS int64)),
                SUM((SELECT SUM(value) FROM UNNEST(mozfun.hist.extract({select_expr}).values)))
            )"""  # noqa
