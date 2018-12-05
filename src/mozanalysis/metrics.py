# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pyspark.sql.functions as F
from functools import partial


CONSTANTS = {
    "ticks_per_second": "5",
    "seconds_per_hour": "3600",
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


class MetricDefinition(object):
    """
    Base class for a `MetricDefinition`.

    Subclasses should define the following fields:

    name (str) : The name of the metric.
    daily_columns (list of str) : A list of columns, as `str`, used to select
        the columns on which to perform the per-day aggregations.
    daily_aggregations (list of `pyspark.sql.functions`) : A list of
        `pyspark.sql.functions`, e.g. `F.sum`, to perform the per-day
        aggregate on the columns defined in `daily_columns`.
    columns (list of str) : A list of columns, as `str`, used to select from
        the initial data set.
    aggregations (list of `pyspark.sql.functions`) : A list of
        `pyspark.sql.functions`, e.g. `F.sum`, to perform the per-client
        aggregate on the columns defined in `columns`.
    final_expression (`pyspark.sql.function.expr`) : An expression used as a
        final computation against the aggregated columns.

    Subclasses can optionally define the following:

    stats (list of executables) : A list of executables passed to `bootstrap`
        for computing the statistics. The default is stats for the mean, 5th,
        25th, 50th, 75th, and 95th percentiles.

    """

    stats = [np.mean, p5, p25, p50, p75, p95]


sum_total_uris_expr = F.expr(
    "SUM(COALESCE(scalar_parent_browser_engagement_total_uri_count, 0))"
).alias("sum_total_uris")


sum_total_hours_expr = F.expr(
    "SUM(COALESCE(subsession_length, 0) / {seconds_per_hour})".format(**CONSTANTS)
).alias("sum_total_hours")


sum_active_hours_expr = F.expr(
    "SUM(COALESCE(active_ticks, 0) * {ticks_per_second} / {seconds_per_hour})".format(
        **CONSTANTS
    )
).alias("sum_active_hours")


class EngagementAvgDailyHours(MetricDefinition):
    name = "Engagement Avg Daily Hours"
    daily_columns = [F.col("subsession_length")]
    daily_aggregations = [sum_total_hours_expr]
    columns = [F.col("sum_total_hours")]
    aggregations = [F.avg("sum_total_hours").alias("daily_hours")]
    final_expression = F.col("daily_hours")


class EngagementAvgDailyActiveHours(MetricDefinition):
    name = "Engagement Avg Daily Active Hours"
    daily_columns = [F.col("active_ticks")]
    daily_aggregations = [sum_active_hours_expr]
    columns = [F.col("sum_active_hours")]
    aggregations = [F.avg("sum_active_hours").alias("active_hours")]
    final_expression = F.col("active_hours")


class EngagementHourlyUris(MetricDefinition):
    name = "Engagement URIs per Active Hour"
    daily_columns = [
        F.col("scalar_parent_browser_engagement_total_uri_count"),
        F.col("active_ticks"),
    ]
    daily_aggregations = [sum_total_uris_expr, sum_active_hours_expr]
    columns = [F.col("sum_total_uris"), F.col("sum_active_hours")]
    aggregations = [
        F.expr(
            "SUM(sum_total_uris) / ({hours_epsilon} + SUM(sum_active_hours))".format(
                **CONSTANTS
            )
        ).alias("hourly_uris")
    ]
    final_expression = F.col("hourly_uris")


class EngagementIntensity(MetricDefinition):
    name = "Engagement Intensity"
    daily_columns = [F.col("subsession_length"), F.col("active_ticks")]
    daily_aggregations = [sum_total_hours_expr, sum_active_hours_expr]
    columns = [F.col("sum_total_hours"), F.col("sum_active_hours")]
    aggregations = [
        F.expr(
            "SUM(sum_active_hours) / ({hours_epsilon} + SUM(sum_total_hours))".format(
                **CONSTANTS
            )
        ).alias("intensity")
    ]
    final_expression = F.col("intensity")
