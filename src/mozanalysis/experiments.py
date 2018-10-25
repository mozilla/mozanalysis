# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import pyspark.sql.functions as F

from mozanalysis.stats import bootstrap
from mozanalysis.utils import dedupe_columns


class ExperimentAnalysis(object):
    def __init__(self, dataset):
        self.sc = dataset._sc
        self._dataset = dataset
        self._metrics = None
        self._date_aggregate_by = "submission_date_s3"
        self._aggregate_by = "client_id"
        self._split_by = "experiment_branch"

    def metrics(self, *metrics):
        self._metrics = metrics
        return self

    def date_aggregate_by(self, column):
        self._date_aggregate_by = column
        return self

    def aggregate_by(self, column):
        self._aggregate_by = column
        return self

    def split_by(self, column):
        self._split_by = column
        return self

    def run(self):
        """
        Run the full analysis on the provided dataset.

        This will perform the per-client per-day aggregation, as well as the
        analysis aggregations, across all metrics.

        """
        df = self.aggregate_per_client_daily(self._dataset)
        return self.analyze(df)

    def get_split_by_values(self, dataset):
        """
        Returns the dictinct column values of the `split_by` column.

        """
        return [
            r[self._split_by]
            for r in dataset.select(self._split_by).distinct().collect()
        ]

    def aggregate_per_client_daily(self, dataset):
        """
        Returns a pyspark dataframe of the data aggregated by the column
        defined in the `date_aggregate_by` call, which is
        "submission_date_s3" by default.

        """
        cols = [
            F.col(self._aggregate_by),
            F.col(self._split_by),
            F.col(self._date_aggregate_by),
        ]
        aggs = []
        for m in self._metrics:
            cols.extend(m.daily_columns)
            aggs.extend(m.daily_aggregations)

        cols = dedupe_columns(cols)
        aggs = dedupe_columns(aggs)

        df = (
            dataset.select(*cols)
            .groupBy(self._aggregate_by, self._split_by, self._date_aggregate_by)
            .agg(*aggs)
        )
        return df

    def analyze(self, dataset):
        """
        Returns a panda dataframe for the provided metrics.

        This also splits the dataset into multiple groups using the column
        defined in the `split_by` call, which is "experiment_branch" by
        default.

        Arguments:

        - dataset (a pyspark Dataframe) : Expects a dataframe with the columns
            defined in the `MetricDefinition`s `columns` field.

        """
        dataset.cache()

        data = []
        branches = self.get_split_by_values(dataset)

        for m in self._metrics:
            # TODO: Use a lib or make this more robust.
            metric_name = m.name.replace(" ", "_").lower()

            agg_df = (
                dataset.select(*([self._aggregate_by, self._split_by] + m.columns))
                .groupBy(self._aggregate_by, self._split_by)
                .agg(*m.aggregations)
                .select(
                    *[
                        self._aggregate_by,
                        self._split_by,
                        m.final_expression.alias(metric_name),
                    ]
                )
            )
            for branch in branches:
                for stat in m.stats:
                    bs = bootstrap(
                        self.sc,
                        agg_df.filter(F.col(self._split_by) == branch)
                        .select(metric_name)
                        .collect(),
                        stat,
                    )
                    data.append(
                        {
                            "branch": branch,
                            "metric_name": metric_name,
                            "stat_name": stat.__name__,
                            "stat_value": bs["calculated_value"],
                            "ci_low": bs["confidence_low"],
                            "ci_high": bs["confidence_high"],
                        }
                    )

        dataset.unpersist()

        return pd.DataFrame(
            data,
            columns=[
                "branch",
                "metric_name",
                "stat_name",
                "stat_value",
                "ci_low",
                "ci_high",
            ],
        )
