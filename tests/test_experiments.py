# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from itertools import product

import mock
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

# NOTE: The metrics are not imported here b/c they are evaluated and require a
# spark context, which isn't available outside the test functions.
from mozanalysis.experiments import ExperimentAnalysis


def test_chaining():
    dataset = mock.Mock()
    dataset._sc = "sc"
    exp = (
        ExperimentAnalysis(dataset)
        .metrics("metric1", "metric2")
        .date_aggregate_by("other_date_column")
        .split_by("split")
    )
    assert exp._date_aggregate_by == "other_date_column"
    assert exp._aggregate_by == "client_id"  # The default.
    assert exp._metrics == ("metric1", "metric2")
    assert exp._split_by == "split"


def _generate_data(spark, client_branches):
    """
    Given a dict of clients->branches, generate predictive data for testing.
    """
    submission_date = "2018010%d"  # Fill in date with digits 1-9.

    incrementer = 1
    data_rows = []
    for client in sorted(client_branches.keys()):
        for branch in sorted(client_branches.values()):
            for date in range(1, 10):
                data_rows.append(
                    [
                        client,
                        submission_date % date,
                        "mozanalysis-tests",  # experiment_id
                        client_branches[client],  # experiment_branch
                        2 * incrementer,  # URI counts
                        100 * incrementer,  # subsession length
                        150 * incrementer,  # active ticks
                    ]
                )
                incrementer += 1

    return spark.createDataFrame(
        data_rows,
        [
            "client_id",
            "submission_date_s3",
            "experiment_id",
            "experiment_branch",
            "scalar_parent_browser_engagement_total_uri_count",
            "subsession_length",
            "active_ticks",
        ],
    )


def test_split_by_values(spark):
    # Test that we get the correct branches back.
    df = _generate_data(
        spark, {"aaaa": "control", "bbbb": "variant", "cccc": "control"}
    )
    branches = ExperimentAnalysis(df).get_split_by_values(df)
    assert sorted(branches) == ["control", "variant"]


def test_aggregate_per_client_daily(spark):
    # Test the daily aggregation returns 1 row per date.
    from mozanalysis.metrics import EngagementAvgDailyHours

    df = _generate_data(spark, {"aaaa": "control", "bbbb": "variant"})
    agg_df = (
        ExperimentAnalysis(df)
        .metrics(EngagementAvgDailyHours)
        .aggregate_per_client_daily(df)
    )
    # Assert we have the expected columns.
    assert sorted(agg_df.columns) == sorted(
        ["client_id", "experiment_branch", "submission_date_s3", "sum_total_hours"]
    )
    # Assert the original dataframe has multiple rows per client per day.
    assert (
        df.filter(F.col("submission_date_s3") == "20180101")
        .filter(F.col("client_id") == "aaaa")
        .count()
        == 2
    )
    # Assert we have 1 row per client per day after.
    assert (
        agg_df.filter(F.col("submission_date_s3") == "20180101")
        .filter(F.col("client_id") == "aaaa")
        .count()
        == 1
    )
    # Spot check the value produced.
    expected = (
        (
            df.filter(F.col("client_id") == "aaaa")
            .filter(F.col("submission_date_s3") == "20180109")
            .agg(F.expr("SUM(subsession_length / 3600)").alias("sum_total_hours"))
        )
        .first()
        .asDict()["sum_total_hours"]
    )
    actual = (
        (
            agg_df.filter(F.col("client_id") == "aaaa").filter(
                F.col("submission_date_s3") == "20180109"
            )
        )
        .first()
        .asDict()["sum_total_hours"]
    )
    assert actual == expected


def test_engagement_metrics(spark):
    # Testing all engagement metrics in one pass to reduce amount of Spark testing time.
    from mozanalysis.metrics import (
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
    )

    # Only compute the means across all stats.
    metrics = [
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
    ]
    # Only calculate the means to reduce bootstrap time during testing.
    for m in metrics:
        m.stats = [np.mean]

    df = _generate_data(spark, {"aaaa": "control", "bbbb": "variant"})
    pdf = ExperimentAnalysis(df).metrics(*metrics).run()
    lookup = pdf.set_index(["metric_name", "branch", "stat_name"])["stat_value"]

    def check_stat(metric, control, variant):
        assert np.allclose(lookup.loc[(metric, "control", "mean")], control)
        assert np.allclose(lookup.loc[(metric, "variant", "mean")], variant)

    # sum(subsession lengths / 3600) / n_days
    check_stat("engagement_avg_daily_hours", control=4.75 / 9, variant=13.75 / 9)
    # sum(active ticks * 5 / 3600) / n_days
    check_stat(
        "engagement_avg_daily_active_hours", control=35.625 / 9, variant=103.125 / 9
    )
    # sum(uris) / (1/3600 + sum(active hours))
    check_stat(
        "engagement_uris_per_active_hour",
        control=342 / (1 / 3600.0 + 35.625),
        variant=990 / (1 / 3600.0 + 103.125),
    )
    # sum(active hours) / (1/3600 + sum(total hours))
    check_stat(
        "engagement_intensity",
        control=35.625 / (1 / 3600.0 + 4.75),
        variant=103.125 / (1 / 3600.0 + 13.75),
    )


def test_metrics_handle_nulls(spark):
    from mozanalysis.metrics import (
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
        p50,
    )

    metrics = [
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
    ]
    for m in metrics:
        m.stats = [np.mean, p50]

    data = {
        "client_id": ["a", "b", "c"],
        "experiment_branch": ["control"] * 3,
        "submission_date_s3": ["1", "2", "3"],
        "scalar_parent_browser_engagement_total_uri_count": [None, 10, 10],
        "subsession_length": [10, None, 10],
        "active_ticks": [10, 10, None],
    }
    sdf = spark.createDataFrame(pd.DataFrame(data, dtype=object))
    summary = ExperimentAnalysis(sdf).metrics(*metrics).run()
    # assert that each stat is defined for each metric
    must_have = pd.DataFrame(
        [
            {"stat_name": stat, "metric_name": metric_name}
            for stat, metric_name in product(
                ["mean", "p50"], [m.name.replace(" ", "_").lower() for m in metrics]
            )
        ]
    )
    assert len(must_have.merge(summary, how="inner")) == len(must_have)
    assert not summary.isnull().any(axis=None)
