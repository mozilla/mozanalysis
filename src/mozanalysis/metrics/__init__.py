# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr


@attr.s(frozen=True, slots=True)
class DataSource(object):
    name = attr.ib(validator=attr.validators.instance_of(str))
    from_expr = attr.ib(validator=attr.validators.instance_of(str))
    experiments_column_type = attr.ib(default='simple', type=str)
    client_id_column = attr.ib(default='client_id', type=str)
    submission_date_column = attr.ib(default='submission_date', type=str)

    @property
    def experiments_column_expr(self):
        if self.experiments_column_type is None:
            return ''

        elif self.experiments_column_type == 'simple':
            return """AND (
                    ds.{submission_date} != e.enrollment_date
                    OR `moz-fx-data-shared-prod.udf.get_key`(
                        ds.experiments, '{experiment_slug}'
                    ) IS NOT NULL
                )"""

        elif self.experiments_column_type == 'native':
            return """AND (
                    ds.{submission_date} != e.enrollment_date
                    OR `moz-fx-data-shared-prod.udf.get_key`(
                        ds.experiments, '{experiment_slug}'
                    ).branch IS NOT NULL
            )"""

        elif self.experiments_column_type == 'glean':
            raise NotImplementedError

        else:
            raise ValueError

    def build_query(self, metric_list, time_limits, experiment_slug):
        """Return a nearly-self contained SQL query.

        This query does not define ``enrollments`` but otherwise could
        be executed to query all metrics from this data source.
        """
        return """SELECT
            e.client_id,
            e.analysis_window_start,
            e.analysis_window_end,
            {metrics}
        FROM enrollments e
            LEFT JOIN {from_expr} ds
                ON ds.{client_id} = e.client_id
                AND ds.{submission_date} BETWEEN '{fddr}' AND '{lddr}'
                AND ds.{submission_date} BETWEEN
                    DATE_ADD(e.enrollment_date, interval e.analysis_window_start day)
                    AND DATE_ADD(e.enrollment_date, interval e.analysis_window_end day)
                {ignore_pre_enroll_first_day}
        GROUP BY e.client_id, e.analysis_window_start, e.analysis_window_end""".format(
            client_id=self.client_id_column,
            submission_date=self.submission_date_column,
            from_expr=self.from_expr,
            fddr=time_limits.first_date_data_required,
            lddr=time_limits.last_date_data_required,
            metrics=',\n            '.join(
                "{se} AS {n}".format(
                    se=m.select_expr.format(experiment_slug=experiment_slug), n=m.name
                )
                for m in metric_list
            ),
            ignore_pre_enroll_first_day=self.experiments_column_expr.format(
                submission_date=self.submission_date_column,
                experiment_slug=experiment_slug,
            )
        )

    def get_sanity_metrics(self, experiment_slug):
        if self.experiments_column_type is None:
            return []

        elif self.experiments_column_type == 'simple':
            return [
                Metric(
                    name=self.name + '_has_contradictory_branch',
                    data_source=self,
                    select_expr=agg_any("""`moz-fx-data-shared-prod.udf.get_key`(
                ds.experiments, '{experiment_slug}'
            ) != e.branch"""),
                ),
                Metric(
                    name=self.name + '_has_non_enrolled_data',
                    data_source=self,
                    select_expr=agg_any("""`moz-fx-data-shared-prod.udf.get_key`(
                ds.experiments, '{experiment_slug}'
            ) IS NULL""".format(experiment_slug=experiment_slug))
                ),
            ]

        elif self.experiments_column_type == 'native':
            return [
                Metric(
                    name=self.name + '_has_contradictory_branch',
                    data_source=self,
                    select_expr=agg_any("""`moz-fx-data-shared-prod.udf.get_key`(
                ds.experiments, '{experiment_slug}'
            ).branch != e.branch"""),
                ),
                Metric(
                    name=self.name + '_has_non_enrolled_data',
                    data_source=self,
                    select_expr=agg_any("""`moz-fx-data-shared-prod.udf.get_key`(
                ds.experiments, '{experiment_slug}'
            ).branch IS NULL""".format(experiment_slug=experiment_slug))
                ),
            ]

        elif self.experiments_column_type == 'glean':
            raise NotImplementedError

        else:
            raise ValueError


@attr.s(frozen=True, slots=True)
class Metric(object):
    """Represents an experiment metric.

    Needs to be combined with an analysis window to be measurable!
    """
    name = attr.ib(type=str)
    data_source = attr.ib(type=DataSource)
    select_expr = attr.ib(type=str)


def agg_sum(select_expr):
    """Return a SQL fragment for the sum over the data, with 0-filled nulls.
    """
    return "COALESCE(SUM({}), 0)".format(select_expr)


def agg_any(select_expr):
    """Return the logical OR, with FALSE-filled nulls."""
    return "COALESCE(LOGICAL_OR({}), FALSE)".format(select_expr)
