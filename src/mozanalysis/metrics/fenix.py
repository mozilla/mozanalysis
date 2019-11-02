# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pyspark.sql import functions as F
from pyspark.sql import types as T

from mozanalysis.metrics import Metric, DataSource
from mozanalysis.utils import all_


@DataSource.from_func()
def fenix_baseline(spark, experiment):
    bs = spark.table('org_mozilla_fenix_baseline_parquet')
    return bs.withColumn(
        'client_id', bs.client_info.client_id
    ).withColumn(
        'b_dur', bs.metrics.timespan['glean.baseline.duration']
    ).withColumn(
        'duration', F.col('b_dur.value')
    ).drop('b_dur')


@DataSource.from_func()
def fenix_metrics(spark, experiment):
    met = spark.table('org_mozilla_fenix_metrics_parquet')
    return met.m.withColumn('client_id', met.client_info.client_id)


@DataSource.from_func()
def fenix_events(spark, experiment):
    ev = spark.table('org_mozilla_fenix_events_parquet')
    return ev.select(
        ev.client_info.client_id.alias('client_id'),
        ev.submission_date_s3,
        F.explode_outer(e.events).alias('event')
    ).select(
        'event.*',
        '*'
    ).select(
        '*',
        F.explode_outer(F.col('extra'))
    ).drop('event', 'extra')


@Metric.from_func(fenix_metrics)
def uri_count(met):
    return met.agg_sum(met.metrics.counter['events.total_uri_count'])


@Metric.from_func(fenix_baseline)
def duration(bs):
    return bs.agg_sum(bs.duration)


@Metric.from_func(fenix_events)
def user_reports_site_issue(ev):
    return F.sum(all_([
        ev.name == 'browser_menu_action',
        ev.key == 'item',
        ev.value == 'report_site_issue'
    ]).astype('int'))


@F.udf(returnType=T.IntegerType())
def extract_search_counts(x):
    counts = 0
    if x is not None:
        counts = sum(x.values())
    return counts


@Metric.from_func(fenix_metrics)
def search_count(met):
    return met.agg_sum(
        extract_search_counts(met.metrics.labeled_counter['metrics.search_count'])
    )
