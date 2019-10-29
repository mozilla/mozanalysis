# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from pyspark.sql import functions as F

from mozanalysis.metrics import Metric, DataSource
from mozanalysis.utils import all_, any_


@DataSource.from_func()
def firetv_events(spark, experiment):
    te = spark.table('telemetry_mobile_event_parquet')
    return te.filter(
        te.app_name == 'FirefoxForFireTV'
    ).select(
      te.submission_date_s3,
      te.client_id,
      F.explode_outer(te.events).alias('event')
    )


@Metric.from_func(firetv_events)
def user_show_menus(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'user_show',
        fe.event.object == 'menu',
    ]).astype('int')),


@Metric.from_func(firetv_events)
def home_tile_clicks(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'click',
        fe.event.object == 'home_tile',

        # Otherwise youtube is double counted (per liuche 2019/06/07) :'(
        fe.event.value != 'youtube_tile'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def anything_but_youtube_tile_clicks(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'click',
        fe.event.object == 'home_tile',
        fe.event.value != 'youtube_tile',
        (
            F.isnull(fe.event.extra['tile_id']) |
            (fe.event.extra['tile_id'] != 'youtube')
        )
    ]).astype('int')),


@Metric.from_func(firetv_events)
def bundled_non_youtube_tile_clicks(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'click',
        fe.event.object == 'home_tile',
        fe.event.value == 'bundled',
        fe.event.extra['tile_id'] != 'youtube',
    ]).astype('int')),


@Metric.from_func(firetv_events)
def pocket_video_clicks(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'click',
        fe.event.object == 'menu',
        fe.event.value == 'pocket_video_tile'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def type_urls(fe):
    return F.sum(all_([
            fe.event.category == 'action',
            fe.event.method == 'type_url',
            fe.event.object == 'search_bar',
        ]).astype('int')),


@Metric.from_func(firetv_events)
def type_queries(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'type_query',
        fe.event.object == 'search_bar',
    ]).astype('int')),


@Metric.from_func(firetv_events)
def navigates_or_clicks_not_youtube(fe):
    return F.sum(any_([
        all_([
            fe.event.category == 'action',
            fe.event.method == 'click',
            fe.event.object == 'home_tile',
            fe.event.value != 'youtube_tile',
            (
                F.isnull(fe.event.extra['tile_id']) |
                (fe.event.extra['tile_id'] != 'youtube')
            )
        ]),
        all_([
            fe.event.category == 'action',
            fe.event.method == 'type_url',
            fe.event.object == 'search_bar',
        ]),
        all_([
            fe.event.category == 'action',
            fe.event.method == 'type_query',
            fe.event.object == 'search_bar',
        ])
    ]).astype('int')),


@Metric.from_func(firetv_events)
def browser_backs(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'click',
        fe.event.object == 'menu',
        fe.event.value == 'back'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def remote_backs(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'page',
        fe.event.object == 'browser',
        fe.event.value == 'back'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def tracking_protection_toggle_on(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'change',
        fe.event.object == 'turbo_mode',
        fe.event.value == 'on'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def tracking_protection_toggle_off(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'change',
        fe.event.object == 'turbo_mode',
        fe.event.value == 'off'
    ]).astype('int')),


@Metric.from_func(firetv_events)
def tracking_protection_toggle(fe):
    return F.sum(all_([
        fe.event.category == 'action',
        fe.event.method == 'change',
        fe.event.object == 'turbo_mode',
    ]).astype('int')),
