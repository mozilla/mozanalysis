from pyspark.sql import Row
from pyspark.sql import functions as F


def generate_histogram(spark):
    return spark.createDataFrame([
        Row(
            client_id="a",
            my_histogram={"0": 5, "10": 10, "42": 1, "43": 0},
        ),
        Row(
            client_id="b",
            my_histogram={"0": 10, "10": 0, "42": 38, "43": 5},
        )
    ])


def test_histogram_mean(spark):
    from mozanalysis.udf import histogram_mean
    df = (
        generate_histogram(spark)
        .withColumn("mean", histogram_mean("my_histogram"))
        .toPandas()
        .set_index("client_id")
    )
    assert df.loc["b", "mean"] > df.loc["a", "mean"]


def test_histogram_count(spark):
    from mozanalysis.udf import histogram_count
    df = (
        generate_histogram(spark)
        .withColumn("count", histogram_count("my_histogram"))
        .toPandas()
        .set_index("client_id")
    )
    assert df.loc["a", "count"] == 16
    assert df.loc["b", "count"] == 53


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
            F.explode(sdf.my_histogram).alias("bucket", "count"),
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
