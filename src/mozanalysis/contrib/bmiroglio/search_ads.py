# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pyspark.sql.functions as F


def get_ad_ctr(sdf, groupers):
    """
    Calculates the sum of ad impressions, the sum of ad clicks
    and the click through rate (ctr) of ads for a given spark
    dataframe, grouped by a user-specificed set of columns

    Args
    sdf: a spark dataframe
    groupers: a list of colname names

    Returns a spark dataframe with the columns specified
    in the <groupers> arg along with:
    scalar_parent_browser_search_with_ads_sum,
    scalar_parent_browser_search_ad_clicks_sum,
    ctr
    """

    impressions_str = "scalar_parent_browser_search_with_ads"
    clicks_str = "scalar_parent_browser_search_ad_clicks"

    def explode_ad_map(metric):
        return (
          sdf
          .select("*", F.explode(metric))
          .groupby(groupers + [F.col("key").alias("engine")])
          .agg(F.coalesce(F.sum("value"), F.lit(0)).alias(metric + "_sum"))
        )

    impressions = explode_ad_map(impressions_str)
    clicks = explode_ad_map(clicks_str)

    return (
        impressions
        .join(clicks, on=groupers + ["engine"])
        .withColumn("ctr", F.col(clicks_str + '_sum') / F.col(impressions_str + '_sum'))
    )
