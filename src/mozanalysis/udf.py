from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, LongType, StructField, IntegerType

import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


@udf(LongType())
def histogram_sum(values):
    """Returns the weighted sum of the histogram.

    Note that this is the sum *post*-quantization, which amounts to a
    left-hand-rule discrete integral of the histogram. It is therefore
    likely to be an underestimate of the true sum.
    """
    if values is None:
        return None
    return sum(int(k) * v for k, v in values.items())


@udf(DoubleType())
def histogram_mean(values):
    """Returns the mean of values in a histogram.

    This mean relies on the sum *post*-quantization, which amounts to a
    left-hand-rule discrete integral of the histogram. It is therefore
    likely to be an underestimate of the true mean.
    """
    if values is None:
        return None
    numerator = 0
    denominator = 0
    for k, v in values.items():
        numerator += int(k) * v
        denominator += v
    return numerator / float(denominator)


@udf(LongType())
def histogram_count(values):
    "Counts the number of events recorded by a histogram."
    if values is None:
        return None
    return sum(values.values())


def generate_quantile_udf(sdf, grouping_fields, bucket_field, count_field, quantiles):
    """Returns a UDF for computing histogram quantiles.

    Parameters:
        sdf: Spark DataFrame
        grouping_fields: list of strings representing columns from sdf
        bucket_field: Name of the field containing histogram buckets
        count_field: Name of the field containing counts
        quantiles: List of quantiles as floating point values on [0, 1]

    `sdf` and `grouping_fields` are necessary to infer the schema of the
    returned Spark DataFrame.

    For example, to compute per-user median FX_NEW_WINDOW_MS values for a recent
    sample of main_summary, you could write:

    ---
    import pyspark.sql.functions as F
    ms = spark.table("main_summary")

    grouping_fields = ["client_id"]

    quantile_udf = generate_quantile_udf(
        sdf=ms,
        grouping_fields=grouping_fields,
        bucket_field="bucket",
        count_field="count",
        quantiles=[0.5, 0.95],
    )

    quantiles = (
        ms
        .filter(ms.submission_date_s3 == "20181001")
        .filter(ms.sample_id == "42")
        .select(
            F.explode(ms.histogram_content_fx_new_window_ms).alias("bucket", "count"),
            *grouping_fields
        )
        .groupBy(*grouping_fields)
        .apply(quantile_udf)
        .toPandas()
    )
    ---
    Note that the `quantile_udf` returned by this function is used together
    with the `.apply` method of the grouped DataFrame.

    This will return a table `quantiles` that looks like:

    client_id | quantile | value
    -----------------------------
    id_1      | 0.5      | 2
    id_1      | 0.95     | 23
    id_2      | 0.5      | 4
    ...       | ...      | ...
    """
    schema = (
        sdf
        .select(*grouping_fields)
        .schema
        .add(StructField("quantile", DoubleType(), False))
        .add(StructField("value", DoubleType(), True))
    )

    @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
    def udf(df):
        stats = DescrStatsW(
            df[bucket_field].astype(float), df[count_field]
        ).quantile(quantiles)
        stats = stats.rename_axis("quantile").reset_index(name="value").assign(_dummy=1)
        # Join the (constant) grouping variables back onto the result
        grouping_variables = df[grouping_fields].iloc[:1].assign(_dummy=1)
        result = grouping_variables.merge(stats, on="_dummy").drop("_dummy", axis=1)
        return result

    return udf


def generate_threshold_udf(sdf, grouping_fields, bucket_field, count_field, thresholds):
    """Returns a UDF for calculating the fraction of values in a histogram
    equal to or exceeding a threshold.

    Parameters:
        sdf: Spark DataFrame
        grouping_fields: list of strings representing columns from sdf
        bucket_field: Name of the field containing histogram buckets
        count_field: Name of the field containing counts
        thresholds: List of thresholds. Should probably align on the left edge
            of a bucket.

    `sdf` and `grouping_fields` are necessary to infer the schema of the
    returned Spark DataFrame.

    For example, to compute the per-user fractions of CONTENT_FRAME_TIME values
    equal to or exceeding 103% or 192% of a vsync from a sample of main_summary,
    you could write:

    ---
    import pyspark.sql.functions as F
    ms = spark.table("main_summary")

    grouping_fields = ["client_id"]

    threshold_udf = generate_threshold_udf(
        sdf=ms,
        grouping_fields=grouping_fields,
        bucket_field="bucket",
        count_field="count",
        thresholds=[103, 192],
    )

    thresholded = (
        ms
        .filter(ms.submission_date_s3 == "20181001")
        .filter(ms.sample_id == "42")
        .select(
            F.explode(ms.histogram_gpu_content_frame_time).alias("bucket", "count"),
            *grouping_fields
        )
        .groupBy(*grouping_fields)
        .apply(threshold_udf)
        .toPandas()
    )
    ---
    Note that the `threshold_udf` returned by this function is used together
    with the `.apply` method of the grouped DataFrame.

    This will return a table `thresholded` that looks like:

    client_id | threshold | fraction_exceeding
    ------------------------------------------
    id_1      | 103       | 0.994
    id_1      | 192       | 0.036
    id_2      | 103       | 0.978
    ...       | ...       | ...
    """
    schema = (
        sdf
        .select(*grouping_fields)
        .schema
        .add(StructField("threshold", IntegerType(), False))
        .add(StructField("fraction_exceeding", DoubleType(), True))
    )

    @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
    def udf(df):
        rows = []
        all_sum = float(df[count_field].sum())
        for tx in thresholds:
            fraction = (
                df.loc[df[bucket_field].astype(int) >= tx, count_field].sum() / all_sum
            )
            rows.append({"threshold": tx, "fraction_exceeding": fraction})
        data = pd.DataFrame(rows).assign(_dummy=1)
        grouping_variables = df[grouping_fields].iloc[:1].assign(_dummy=1)
        result = grouping_variables.merge(data, on="_dummy").drop("_dummy", axis=1)
        return result

    return udf
