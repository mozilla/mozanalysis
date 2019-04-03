# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from mozanalysis.stats import bootstrap
from pyspark.sql import Window


class CoreMetrics(Enum):
    ENGAGEMENT = 1
    RETENTION = 2


BASE_COLS = ["client_id", "experiment_branch", "submission_date"]
AGG_COLS = ["client_id", "experiment_branch"]
COLUMNS = {
    CoreMetrics.ENGAGEMENT: [
        "subsession_length",
        "active_ticks",
        "scalar_parent_browser_engagement_total_uri_count",
    ],
    CoreMetrics.RETENTION: [
        "subsession_length",
        "scalar_parent_browser_engagement_total_uri_count",
    ],
}

CONSTANTS = {
    "ticks_per_second": "5",
    "seconds_per_hour": "3600",
    "hourly_uris_considered_active": "5",
    "hours_epsilon": "1 / 3600",  # Used to avoid division by zero.
}

p5 = partial(np.percentile, q=5)
p5.__name__ = "p5"

p25 = partial(np.percentile, q=25)
p25.__name__ = "p25"

p50 = partial(np.percentile, q=50)
p50.__name__ = "p50"

p75 = partial(np.percentile, q=75)
p75.__name__ = "p75"

p95 = partial(np.percentile, q=95)
p95.__name__ = "p95"


class ExperimentAnalysis(object):
    def __init__(self, spark_session, experiment_id, stats=None):
        self.spark_session = spark_session
        self.experiment_id = experiment_id
        self._df = None

        if stats:
            if isinstance(stats, (list, tuple)):
                self.stats = stats
            else:
                self.stats = [stats]
        else:
            self.stats = [np.mean, p5, p25, p50, p75, p95]

    @property
    def df(self):
        if self._df:
            return self._df

        return self.get_data()

    @df.setter
    def df(self, value):
        self._df = value

    def get_experiment_data(self):
        df = self.spark_session.read.parquet("s3://telemetry-parquet/experiments/v1")
        df = df.filter(F.col("experiment_id") == self.experiment_id)
        return df

    def aggregate_per_day(self, df, core_metric=None):
        # If user wants to override outlier percentiles, set it False here and
        # call it explicitly.

        if core_metric is None:
            columns = list(set(sum(COLUMNS.values(), [])))
        else:
            columns = COLUMNS[core_metric]

        df = (
            df.select(
                "client_id",
                "experiment_branch",
                F.date_format(
                    F.to_date("submission_date_s3", format="yyyyMMdd"), "yyyy-MM-dd"
                ).alias("submission_date"),
                *columns
            )
            .filter(F.col("experiment_branch").isNotNull())
            .groupBy(*BASE_COLS)
            # This aggregation simply SUMs the metrics per client per branch
            # per day. Note: This may not work for future metrics that aren't
            # simple scalars.
            .agg(
                *[
                    F.expr("SUM(COALESCE({}, 0))".format(col)).alias(col)
                    for col in columns
                ]
            )
        )

        return df

    def get_data(self, trim_outliers=True, core_metric=None):
        df = self.get_experiment_data()
        df = self.aggregate_per_day(df, core_metric)

        # TODO: Cache this dataframe.

        self._df = df
        return df

    def trim_outliers(self, outlier_percentile=0.9999, relative_error=0.0001):
        # TODO
        # Updates `self._df` and returns it.
        return self._df

    def engagement(self):
        # Compute only the engagement metrics.
        # Return pandas dataframe with data and CIs.
        if not self._df:
            self._df = self.get_data(metric=CoreMetrics.ENGAGEMENT)

        # Get the daily averages for the columns, aggregating per client per
        # branch per day.
        agg_df = (
            self._df.select(*(BASE_COLS + COLUMNS[CoreMetrics.ENGAGEMENT]))
            .withColumn(
                "sum_total_hours",
                self._df.subsession_length / CONSTANTS["seconds_per_hour"],
            )
            .withColumn(
                "sum_active_hours",
                self._df.active_ticks
                * CONSTANTS["ticks_per_second"]
                / CONSTANTS["seconds_per_hour"],
            )
            .withColumn(
                "sum_total_uris",
                self._df.scalar_parent_browser_engagement_total_uri_count,
            )
            .groupBy(*AGG_COLS)
            .agg(
                F.avg("sum_total_hours").alias("engagement_daily_hours"),
                F.avg("sum_active_hours").alias("engagement_daily_active_hours"),
                F.expr(
                    "SUM(sum_total_uris) "
                    "/ ({hours_epsilon} + SUM(sum_active_hours))".format(**CONSTANTS)
                ).alias("engagement_hourly_uris"),
                F.expr(
                    "SUM(sum_active_hours) "
                    "/ ({hours_epsilon} + SUM(sum_total_hours))".format(**CONSTANTS)
                ).alias("engagement_intensity"),
            )
        )

        return agg_df

    def retention(self, week=3):
        # Compute only the retention metrics.
        # Return pandas dataframe with data and CIs.
        if not self._df:
            self._df = self.get_data(metric=CoreMetrics.RETENTION)

        retention_df = (
            self._df.withColumn(
                "enrollment_date",
                F.min("submission_date").over(Window().partitionBy("client_id")),
            )
            .withColumn(
                "sum_total_hours",
                self._df.subsession_length / CONSTANTS["seconds_per_hour"],
            )
            .withColumn(
                "sum_total_uris",
                self._df.scalar_parent_browser_engagement_total_uri_count,
            )
            .select(
                "enrollment_date",
                "sum_total_hours",
                "sum_total_uris",
                F.datediff("submission_date", "enrollment_date").alias("nd"),
                F.floor(F.datediff("submission_date", "enrollment_date") / 7).alias(
                    "week"
                ),
                *self._df.columns
            )
            .filter(F.col("nd") >= 0)
        )

        week3_df = (
            retention_df.filter("week={week}".format(week=week))
            .groupBy(*AGG_COLS)
            .agg(
                F.expr("CASE WHEN SUM(sum_total_hours) > 0 THEN 1 ELSE 0 END").alias(
                    "retention"
                ),
                F.expr("MAX(CASE WHEN sum_total_uris >= 5 THEN 1 ELSE 0 END)").alias(
                    "active_retention"
                ),
            )
        )

        return week3_df

    def analyze(self):
        # TODO
        # Compute all metrics.
        #
        # This is a convenience method.
        #
        # The idea would be to make the calls to `engagement()` and `retention`
        # separately, pass those results to `bootstrap`, the concatenate the
        # pandas dataframes returned from bootstrap and return that.
        pass

    def bootstrap(self, df):
        # Iterates over metrics columns and calculates bootstrapped confidence
        # intervals for the data in those columns.
        metrics = list(set(df.columns) - set(AGG_COLS))
        branches = [
            r["experiment_branch"]
            for r in df.select("experiment_branch").distinct().collect()
        ]

        data = []

        for metric in metrics:
            for branch in branches:
                for stat in self.stats:

                    bootstrap_data = (
                        df.select(metric)
                        .filter(df.experiment_branch == branch)
                        .collect()
                    )
                    bs = bootstrap(
                        self.spark_session.sparkContext, bootstrap_data, stat
                    )
                    data.append(
                        {
                            "branch": branch,
                            "metric": metric,
                            "stat": stat.__name__,
                            "value": bs["calculated_value"],
                            "ci_low": bs["confidence_low"],
                            "ci_high": bs["confidence_high"],
                        }
                    )

        return pd.DataFrame(
            data, columns=["branch", "metric", "stat", "value", "ci_low", "ci_high"]
        )
