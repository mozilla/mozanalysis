import attr
import re

from mozanalysis.metrics import DataSource
from mozanalysis.bq import BigQueryContext, sanitize_table_name_for_bq

from metric_config_parser.experiment import Experiment

from textwrap import dedent

import numpy as np
from scipy.special import lambertw
from datetime import datetime

from abc import ABC


class ExperimentAnnotationMissingError(Exception):
    pass


@attr.s(frozen=True, slots=True)
class InflightDataSource(DataSource):
    """
    POC implementation of [this proposal](https://docs.google.com/document/d/1bNTGPDan_ANlKQy6p9Y3o5ZwLk_XyvSb9br3gBUywxA/edit)
    Specifically, Theorem 4.1 from [Design-Based Confidence Sequences for Anytime-valid Causal Inference](https://arxiv.org/pdf/2210.08639.pdf)
    """  # noqa

    timestamp_column = attr.ib(
        default="submission_timestamp", validator=attr.validators.instance_of(str)
    )

    EXPERIMENT_COLUMN_TYPES = (None, "simple", "native", "glean", "main_live")  # noqa

    @property
    def experiments_column_expr(self) -> str:
        """Returns a SQL expression to extract the branch from the
        experiment annotations"""
        if self.experiments_column_type is None:
            raise ExperimentAnnotationMissingError

        elif self.experiments_column_type == "simple":
            return """`mozfun.map.get_key`(ds.experiments, '{experiment_slug}')"""

        elif self.experiments_column_type == "native":
            return (
                """`mozfun.map.get_key`(ds.experiments, '{experiment_slug}').branch"""
            )

        elif self.experiments_column_type == "glean":
            return """`mozfun.map.get_key`(ds.ping_info.experiments, '{experiment_slug}').branch"""

        elif self.experiments_column_type == "main_live":
            return """`mozfun.map.get_key`(ds.environment.experiments, '{experiment_slug}').branch"""

        else:
            raise ValueError

    def render_records_query(
        self,
        metric: "InflightMetric",
        experiment: Experiment,
        from_expr_dataset: str | None = None,
    ) -> str:
        """
        Returns the SQL to create a client-timestamp level dataset.

        This does not assume an enrollments table has been created, instead
        relies upon experiment annotations.
        """

        query = f"""
    SELECT 
        ds.client_id,
        {self.experiments_column_expr.format(experiment_slug=experiment.normandy_slug)} AS branch,
        MIN(ds.{self.timestamp_column}) AS event_timestamp,
        MIN_BY({metric.select_expr.format(experiment_slug=experiment.normandy_slug)}, ds.{self.timestamp_column}) AS {metric.name}
    FROM {self.from_expr_for(from_expr_dataset)} ds
    WHERE 1=1
        AND ds.{self.timestamp_column} BETWEEN "{experiment.start_date.strftime('%Y-%m-%d')}" AND "{(experiment.end_date or datetime.now()).strftime('%Y-%m-%d')}"
        AND {self.experiments_column_expr.format(experiment_slug=experiment.normandy_slug)} IS NOT NULL 
    GROUP BY client_id, branch
    ORDER BY event_timestamp"""  # noqa

        return query


@attr.s(frozen=True, slots=True)
class InflightMetric:
    name = attr.ib(type=str)
    select_expr = attr.ib(type=str)
    data_source = attr.ib(type=InflightDataSource)
    friendly_name = attr.ib(type=str | None, default=None)
    description = attr.ib(type=str | None, default=None)
    app_name = attr.ib(type=str | None, default=None)

    def render_records_query(
        self, experiment: Experiment, from_expr_dataset: str | None = None
    ) -> str:
        return self.data_source.render_records_query(
            self, experiment, from_expr_dataset
        )

    def record_view_name(self, experiment: Experiment) -> str:
        bq_experiment_slug = sanitize_table_name_for_bq(experiment.normandy_slug)
        metric_slug = sanitize_table_name_for_bq(self.name)
        view_name = f"records_{bq_experiment_slug}_{metric_slug}"
        return view_name

    def publish_records_view(
        self,
        context: BigQueryContext,
        experiment: Experiment,
        from_expr_dataset: str | None = None,
    ) -> None:
        view_name = self.record_view_name(experiment)
        view_sql = self.render_records_query(experiment, from_expr_dataset)

        context.create_view(view_name, view_sql)


@attr.s()
class InflightStatistic(ABC):

    alpha: float = 0.05

    def render_statistics_query(
        self,
        experiment: Experiment,
        metric: "InflightMetric",
        **statistics_kwargs,
    ) -> str:
        raise NotImplementedError

    def publish_statistics_view(
        self,
        experiment: Experiment,
        metric: InflightMetric,
        context: BigQueryContext,
        **statistical_kwargs,
    ) -> None:
        raise NotImplementedError


@attr.s()
class DesignBasedConfidenceSequences(InflightStatistic):
    minimum_width_observations: int = 100

    def render_statistics_query_piece_prep(
        self, comparison_branch: str, reference_branch: str, metric_name: str
    ) -> str:
        """
        Prepares/formats the record-level data for the statistical computations.

        Filters to clients from the `reference_branch` or `comparison_branch`,
        constructs treatment indicators, a `Y_i` column, and a rank column.

        Assumes an upstream CTE holding the output of `render_record_query`
        named `records`.
        """
        query = f"""SELECT 
            *,
            CASE WHEN branch = "{comparison_branch}" THEN 1 ELSE 0 END AS treated,
            CASE WHEN branch = "{reference_branch}" THEN 1 ELSE 0 END AS not_treated,
            {metric_name} AS Y_i,
            RANK() OVER (ORDER BY event_timestamp) AS n
        FROM records 
        WHERE branch in ("{reference_branch}", "{comparison_branch}")"""

        return query

    def render_statistics_query_piece_sufficient_statistics(self) -> str:
        """
        Builds upon `render_statistics_query_piece_intro` to add the sufficient statistics
        `tau_hat_i` and `sigma_hat_sq_i` necessary to calculate the confidence sequence.

        Adds:
        - `tau_hat_i`: either +1/2*metric value (in case of comparison branch) or
        -1/2*metric value (in case of reference branch).
        - `sigma_hat_sq_i`: either +1/4*(metric value)^2 (in case of comparison branch) or
        -1/4*(metric value)^2 (in case of reference branch).

        Assumes an upstream CTE holding the output of `render_statistics_query_piece_prep`
        named `prep`.
        """

        query = """SELECT 
            *, 
            treated*Y_i/0.5 - not_treated*Y_i/0.5 AS tau_hat_i,
            treated*POW(Y_i,2)/POW(0.5,2) + not_treated*POW(Y_i,2)/POW(0.5,2) AS sigma_hat_sq_i,
        FROM prep"""

        return query

    def render_statistics_query_piece_accumulators(self) -> str:
        """
        Builds upon `render_statistics_query_piece_sufficient_statistics` to construct
        expanding sufficient statistics (accumulate the sufficient statistics over time).

        Adds:
        - point_est: the expanding average of `tau_hat_i`, over clients present up to and
        including this time point. Under null hypothesis, distribution is centered at 0.
        - var_est: the expanding variance estimator, over clients present up to an including
        this time point. Known as S_n in the literature.

        Assumes an upstream CTE holding the output of
        `render_statistics_query_piece_sufficient_statistics` named `sufficient_statistics`.
        """

        query = """SELECT 
            *, 
            -- SUM(tau_hat_i) OVER (ORDER BY event_timestamp) AS tau_hat_i_acc,
            1/n * SUM(tau_hat_i) OVER (ORDER BY event_timestamp) AS point_est,
            SUM(sigma_hat_sq_i) OVER (ORDER BY event_timestamp) AS var_est
        FROM sufficient_statistics"""

        return query

    def render_statistics_query_piece_ci_terms(self) -> str:
        """
        Builds upon `render_statistics_query_piece_accumulators` to construct
        the two terms needed to calculate the width of the confidence sequence.

        Assumes an upstream CTE holding the output of
        `render_statistics_query_piece_accumulators` named `accumulators`.
        """

        eta_sq = self.eta**2
        alpha_sq = self.alpha**2

        query = f"""SELECT 
            *,
            (var_est * {eta_sq} + 1)/{eta_sq} AS width_term_1,
            LN((var_est * {eta_sq}+1)/{alpha_sq}) AS width_term_2
        FROM accumulators"""

        return query

    def render_statistics_query_piece_ci_width(self) -> str:
        """
        Builds upon `render_statistics_query_piece_accumulators` to construct
        the two terms needed to calculate the width of the confidence sequence.

        Adds:
        - ci_width: the width of the confidence sequence at this time.

        Assumes an upstream CTE holding the output of
        `render_statistics_query_piece_ci_terms` named `ci_terms`.
        """

        query = """SELECT 
            *, 
            (1/n) * SQRT(width_term_1 * width_term_2) AS ci_width
        FROM ci_terms"""

        return query

    def render_statistics_query_piece_cleanup(self, comparison_branch: str) -> str:
        """
        Cleans up the output of `render_statistics_query_piece_ci_width`.

        Assumes an upstream CTE holding the output of
        `render_statistics_query_piece_ci_width` named `ci_width_term`
        """

        query = f"""SELECT 
            event_timestamp,
            n, 
            "{comparison_branch}" AS comparison_branch,
            point_est, 
            point_est - ci_width AS ci_lower,
            point_est + ci_width AS ci_upper
        FROM ci_width_term"""

        return query

    def render_statistics_query_one_branch(
        self,
        comparison_branch: str,
        reference_branch: str,
        metric_name: str,
    ) -> str:
        """
        Builds the statistical query to construct the confidence sequence to compare
        a `comparison_branch` to a `reference_branch`.
        """

        query = f"""
    WITH prep AS (
        {self.render_statistics_query_piece_prep(comparison_branch, reference_branch, metric_name)}
    ), sufficient_statistics AS (
        {self.render_statistics_query_piece_sufficient_statistics()}
    ), accumulators AS (
        {self.render_statistics_query_piece_accumulators()}
    ), ci_terms AS (
        {self.render_statistics_query_piece_ci_terms()}
    ), ci_width_term AS (
        {self.render_statistics_query_piece_ci_width()}
    ), ci_cleanup AS (
        {self.render_statistics_query_piece_cleanup(comparison_branch)}
    )
    SELECT *
    FROM ci_cleanup
"""

        return query

    def render_union_query(
        self, comparison_branches: list[str], full_sample: bool = False
    ) -> str:
        clean_comparison_branches = [
            sanitize_table_name_for_bq(branch) for branch in comparison_branches
        ]
        branch_timestamps = ", ".join(
            [f"{branch}.event_timestamp" for branch in clean_comparison_branches]
        )
        query = f"""
SELECT 
    n,
    LEAST({branch_timestamps}) AS record_timestamp,"""

        for branch in clean_comparison_branches:
            query += f"""
    {branch}.point_est AS point_est_{branch},
    {branch}.ci_lower AS ci_lower_{branch},
    {branch}.ci_upper AS ci_upper_{branch},"""

        query += f"""
FROM {clean_comparison_branches[0]}"""

        if len(clean_comparison_branches) > 1:
            for next_branch in clean_comparison_branches[1:]:
                query += f"""
FULL OUTER JOIN {next_branch} 
USING(n)"""

        query += f"""
ORDER BY record_timestamp"""

        return query

    def render_statistics_query(
        self,
        experiment: Experiment,
        metric: InflightMetric,
        full_sample: bool = False,
        **ignored_kwargs,
    ) -> str:

        metric_view = metric.record_view_name(experiment)

        comparison_branches = [
            branch.slug
            for branch in experiment.branches
            if branch.slug != experiment.reference_branch
        ]

        query = dedent(
            f"""WITH records AS (SELECT * FROM {metric_view}
)"""
        )

        for comparison_branch in comparison_branches:
            comparison_branch_name = sanitize_table_name_for_bq(comparison_branch)
            subquery = self.render_statistics_query_one_branch(
                comparison_branch,
                experiment.reference_branch,
                metric.name,
            )
            query += f""", {comparison_branch_name} AS ({subquery})"""

        query += "\n"
        query += self.render_union_query(comparison_branches, full_sample)

        return query

    def statistics_view_name(
        self, experiment: Experiment, metric: InflightMetric
    ) -> str:
        bq_experiment_slug = sanitize_table_name_for_bq(experiment.normandy_slug)
        metric_slug = sanitize_table_name_for_bq(metric.name)
        statistics_slug = self.name()
        view_name = f"statistics_{bq_experiment_slug}_{metric_slug}_{statistics_slug}"
        return view_name

    def publish_statistics_view(
        self,
        experiment: Experiment,
        metric: InflightMetric,
        context: BigQueryContext,
        full_sample: bool = False,
        **ignored_runtime_statistical_kwargs,
    ) -> None:
        view_name = self.statistics_view_name(experiment, metric)
        view_sql = self.render_statistics_query(experiment, metric, full_sample)

        context.create_view(view_name, view_sql)

    @property
    def eta(self) -> float:
        """
        Returns the `eta` (tuning parameter) that minimizes the relative width of the
        confidence sequence after `minimum_width_observations` clients enrolled. Note
        that each comparison is done with one branch relative to control, so account
        for the number of branches when calculating this term. E.g., a 5-branch experiment
        with 500,000 total enrollees has `eta` defined for `minimum_width_observations`
        in (1,200_000].

        We default to 100 to focus the "alpha spending" near the start of the experiment.
        """
        alpha_sq = self.alpha**2
        eta = np.sqrt(
            (-1 * lambertw(-1 * alpha_sq * np.exp(1), -1) - 1)
            / self.minimum_width_observations
        ).real
        assert np.isfinite(eta)
        return eta

    @classmethod
    def name(cls):
        """Return snake-cased name of the statistic."""
        # https://stackoverflow.com/a/1176023
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class InflightSummary:
    metric: InflightMetric
    statistic: InflightStatistic
    experiment: Experiment

    def publish_views(
        self, context: BigQueryContext, **runtime_statistical_kwargs
    ) -> None:
        self.metric.publish_records_view(context, self.experiment)
        self.statistic.publish_statistics_view(
            self.experiment, self.metric, context, **runtime_statistical_kwargs
        )
