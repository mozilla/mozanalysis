from pyspark.sql import Row
from pyspark.sql import functions as F
from pytest import approx


def generate_histogram(spark):
    return spark.createDataFrame(
        [
            Row(client_id="a", my_histogram={"0": 5, "10": 10, "42": 1, "43": 0}),
            Row(client_id="b", my_histogram={"0": 10, "10": 0, "42": 38, "43": 5}),
        ]
    )


def test_histogram_udfs(spark):
    from mozanalysis.udf import histogram_mean, histogram_count, histogram_sum

    df = (
        generate_histogram(spark)
        .withColumn("mean", histogram_mean("my_histogram"))
        .withColumn("count", histogram_count("my_histogram"))
        .withColumn("sum", histogram_sum("my_histogram"))
        .toPandas()
        .set_index("client_id")
    )
    assert df.loc["a", "count"] == 16
    assert df.loc["b", "count"] == 53
    assert df.loc["a", "sum"] == 142
    assert df.loc["b", "sum"] == 38 * 42 + 5 * 43
    assert df.loc["a", "mean"] == float(df.loc["a", "sum"]) / df.loc["a", "count"]
    assert df.loc["b", "mean"] == float(df.loc["b", "sum"]) / df.loc["b", "count"]


def test_histogram_quantiles(spark):
    from mozanalysis.udf import generate_quantile_udf

    sdf = generate_histogram(spark)
    quantile_udf = generate_quantile_udf(
        sdf=sdf,
        grouping_fields=["client_id"],
        bucket_field="bucket",
        count_field="count",
        quantiles=[0.5, 0.95],
    )

    df = (
        sdf
        .select(
            sdf.client_id,
            F.explode(sdf.my_histogram).alias("bucket", "count")
        )
        .groupBy(sdf.client_id)
        .apply(quantile_udf)
        .toPandas()
        .set_index(["client_id", "quantile"])
    )

    assert df.loc[("a", 0.5), "value"] == 10
    assert df.loc[("a", 0.95), "value"] == 42
    assert df.loc[("b", 0.5), "value"] == 42
    assert df.loc[("b", 0.95), "value"] == 43


def test_histogram_threshold(spark):
    from mozanalysis.udf import generate_threshold_udf

    sdf = generate_histogram(spark)
    threshold_udf = generate_threshold_udf(
        sdf=sdf,
        grouping_fields=["client_id"],
        bucket_field="bucket",
        count_field="count",
        thresholds=[0, 43],
    )

    df = (
        sdf
        .select(
            sdf.client_id,
            F.explode(sdf.my_histogram).alias("bucket", "count")
        )
        .groupBy(sdf.client_id)
        .apply(threshold_udf)
        .toPandas()
        .set_index(["client_id", "threshold"])
    )

    def check(client, threshold, fraction):
        value = df.loc[(client, threshold), "fraction_exceeding"]
        assert value == approx(fraction)

    check("a", 0, 1)
    check("a", 43, 0)
    check("b", 0, 1)
    check("b", 43, 5.0 / 53)
