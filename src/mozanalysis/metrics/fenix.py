# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pyspark.sql import functions as F

from mozanalysis.metrics import Metric, DataSource, agg_sum
from mozanalysis.utils import all_
from mozanalysis.udf import extract_search_counts


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
    return met.withColumn('client_id', met.client_info.client_id)


@DataSource.from_func()
def fenix_events(spark, experiment):
    ev = spark.table('org_mozilla_fenix_events_parquet')
    return ev.select(
        ev.client_info.client_id.alias('client_id'),
        ev.submission_date_s3,
        F.explode_outer(ev.events).alias('event')
    ).select(
        'event.*',
        '*'
    ).select(
        '*',
        F.explode_outer(F.col('extra'))
    ).drop('event', 'extra')


@Metric.from_func(fenix_baseline)
def uri_count(met):
    return agg_sum(met.metrics.counter['events.total_uri_count'])


@Metric.from_func(fenix_baseline)
def duration(bs):
    return agg_sum(bs.duration)


@Metric.from_func(fenix_metrics)
def search_count(met):
    return agg_sum(
        extract_search_counts(met.metrics.labeled_counter['metrics.search_count'])
    )


def calc_num_events(ev, name, key, value):
    return F.sum(all_([
        ev.name == name,
        ev.key == key,
        ev.value == value
    ]).astype('int'))


@Metric.from_func(fenix_events)
def user_reports_site_issue_count(ev):
    return calc_num_events(ev, 'browser_menu_action', 'item', 'report_site_issue')


@Metric.from_func(fenix_events)
def user_reload_count(ev):
    return calc_num_events(ev, 'browser_menu_action', 'item', 'reload')


@Metric.from_func(fenix_baseline)
def baseline_ping_count(bs):
    return F.count(bs.metrics)


@Metric.from_func(fenix_metrics)
def metric_ping_count(met):
    return F.count(met.metrics)


@Metric.from_func(fenix_baseline)
def first_run_date(bs):
    return F.min(bs.first_run_date)
