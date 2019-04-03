# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import datetime
from itertools import product

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from mozanalysis.experiments import COLUMNS, CoreMetrics, ExperimentAnalysis, p50


def _generate_data(spark, client_branches, num_days=9):
    """
    Given a dict of clients->branches, generate predictive data for testing.
    """
    submission_date_base = datetime.date(2018, 1, 1)

    incrementer = 1
    data_rows = []
    for client in sorted(client_branches.keys()):
        for branch in sorted(client_branches.values()):
            for days in range(0, num_days):
                data_rows.append(
                    [
                        client,
                        (submission_date_base + datetime.timedelta(days=days)).strftime(
                            "%Y%m%d"
                        ),
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


def test_aggregate_per_client_daily(spark):
    # Test `get_data` returns 1 row client per day.
    df = _generate_data(spark, {"aaaa": "control", "bbbb": "variant"})
    agg_df = ExperimentAnalysis(
        spark, experiment_id="mozanalysis-tests"
    ).aggregate_per_day(df, CoreMetrics.RETENTION)
    # Assert we have the expected columns.
    assert sorted(agg_df.columns) == sorted(
        ["client_id", "experiment_branch", "submission_date"]
        + COLUMNS[CoreMetrics.RETENTION]
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
        agg_df.filter(F.col("submission_date") == "2018-01-01")
        .filter(F.col("client_id") == "aaaa")
        .count()
        == 1
    )
    # Spot check the value produced.
    expected = (
        (
            df.filter(F.col("client_id") == "aaaa")
            .filter(F.col("submission_date_s3") == "20180109")
            .agg(F.expr("SUM(subsession_length)").alias("subsession_length"))
        )
        .first()
        .asDict()["subsession_length"]
    )
    actual = (
        (
            agg_df.filter(F.col("client_id") == "aaaa").filter(
                F.col("submission_date") == "2018-01-09"
            )
        )
        .first()
        .asDict()["subsession_length"]
    )
    assert actual == expected


def test_engagement_metrics(spark):
    df = _generate_data(spark, {"aaaa": "control", "bbbb": "variant"})

    # Only compute the means across all stats.
    ea = ExperimentAnalysis(spark, experiment_id="mozanalysis-tests", stats=[np.mean])
    df = ea.aggregate_per_day(df, CoreMetrics.ENGAGEMENT)
    ea.df = df
    engagement_df = ea.engagement()
    summary = ea.bootstrap(engagement_df)
    lookup = summary.set_index(["metric", "branch", "stat"])["value"]

    def check_stat(metric, control, variant):
        assert np.allclose(lookup.loc[(metric, "control", "mean")], control)
        assert np.allclose(lookup.loc[(metric, "variant", "mean")], variant)

    # sum(subsession lengths / 3600) / n_days
    check_stat("engagement_daily_hours", control=4.75 / 9, variant=13.75 / 9)
    # sum(active ticks * 5 / 3600) / n_days
    check_stat("engagement_daily_active_hours", control=35.625 / 9, variant=103.125 / 9)
    # sum(uris) / (1/3600 + sum(active hours))
    check_stat(
        "engagement_hourly_uris",
        control=342 / (1 / 3600.0 + 35.625),
        variant=990 / (1 / 3600.0 + 103.125),
    )
    # sum(active hours) / (1/3600 + sum(total hours))
    check_stat(
        "engagement_intensity",
        control=35.625 / (1 / 3600.0 + 4.75),
        variant=103.125 / (1 / 3600.0 + 13.75),
    )


def test_retention_metrics(spark):
    def get_date(days):
        base_day = datetime.date(2018, 1, 1)
        return (base_day + datetime.timedelta(days=days)).strftime("%Y%m%d")
    data = {
        "client_id": ["a", "b"] * 30,
        "experiment_branch": ["control", "variant"] * 30,
        "submission_date_s3": [get_date(d) for d in range(30)]
        + [get_date(d) for d in range(30)],
        "scalar_parent_browser_engagement_total_uri_count": [0, 10] * 30,
        "subsession_length": [0, 10] * 30,
    }
    sdf = spark.createDataFrame(pd.DataFrame(data, dtype=object))
    # Only compute the means across all stats.
    ea = ExperimentAnalysis(spark, experiment_id="mozanalysis-tests", stats=[np.mean])
    df = ea.aggregate_per_day(sdf, CoreMetrics.RETENTION)
    ea.df = df
    retention_df = ea.retention()
    summary = ea.bootstrap(retention_df)
    lookup = summary.set_index(["metric", "branch", "stat"])["value"]

    def check_stat(metric, control, variant):
        assert np.allclose(lookup.loc[(metric, "control", "mean")], control)
        assert np.allclose(lookup.loc[(metric, "variant", "mean")], variant)

    check_stat("retention", control=0.0, variant=1.0)
    check_stat("active_retention", control=0.0, variant=1.0)


def test_metrics_handle_nulls(spark):
    data = {
        "client_id": ["a", "b", "c"],
        "experiment_branch": ["control"] * 3,
        "submission_date_s3": ["1", "2", "3"],
        "scalar_parent_browser_engagement_total_uri_count": [None, 10, 10],
        "subsession_length": [10, None, 10],
        "active_ticks": [10, 10, None],
    }
    sdf = spark.createDataFrame(pd.DataFrame(data, dtype=object))
    ea = ExperimentAnalysis(
        spark, experiment_id="mozanalysis-tests", stats=[np.mean, p50]
    )
    df = ea.aggregate_per_day(sdf, CoreMetrics.ENGAGEMENT)
    ea.df = df
    engagement_df = ea.engagement()
    summary = ea.bootstrap(engagement_df)

    # assert that each stat is defined for each metric
    metrics = [
        "engagement_daily_hours",
        "engagement_daily_active_hours",
        "engagement_hourly_uris",
        "engagement_intensity",
    ]
    must_have = pd.DataFrame(
        [
            {"stat": stat, "metric": metric_name}
            for stat, metric_name in product(["mean", "p50"], metrics)
        ]
    )
    assert len(must_have.merge(summary, how="inner")) == len(must_have)
    assert not summary.isnull().any(axis=None)
