# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pyspark.sql import functions as F

from mozanalysis.metrics import Metric, DataSource, agg_sum, agg_any
from mozanalysis.utils import all_  # , any_
from mozanalysis.udf import pings_histogram_mean, mean_narm


clients_daily = DataSource.from_table_name('clients_daily')
main_summary = DataSource.from_table_name('main_summary')
search_clients_daily = DataSource.from_table_name('search_clients_daily')
events = DataSource.from_table_name('events')


@DataSource.from_func()
def telemetry_shield_study_parquet(spark, experiment):
    """DataSource commonly used with addon studies.

    Used when we need to collect experiment-specific telemetry. We
    filter to just include the data submitted by this experiment's
    addon.
    """
    tssp = spark.table('telemetry_shield_study_parquet')

    this_exp = tssp.filter(
        tssp.payload.study_name == experiment.experiment_slug
    ).withColumnRenamed('submission', 'submission_date_s3')

    if experiment.addon_version is None:
        return this_exp
    else:
        return this_exp.filter(
            tssp.payload.addon_version == experiment.addon_version
        )


# @DataSource.from_func()
# def crash_summary(spark, experiment):
#     cs = spark.table('crash_summary_v2')
#     cs = cs.select(
#         F.regexp_replace(cs.submission_date, '-', '').alias('submission_date_s3'),
#         cs.client_id,
#         'payload.*',
#         cs.experiments)
#     # cs = cs.filter(cs.experiments[experiment.experiment_slug].isNotNull())
#     return cs


@Metric.from_func(clients_daily)
def active_hours(cd):
    """Active hours, from ``active_ticks``

    At any given moment, a client is "active" if there was a keyboard or
    mouse interaction (click, scroll, move) in the previous 5 seconds.
    """
    return agg_sum(cd.active_hours_sum)


@Metric.from_func(search_clients_daily)
def search_count(scd):
    return agg_sum(scd.sap)


@Metric.from_func(search_clients_daily)
def ad_clicks(scd):
    return agg_sum(scd.ad_click)


@Metric.from_func(clients_daily)
def uri_count(cd):
    return agg_sum(cd.scalar_parent_browser_engagement_total_uri_count_sum)


@Metric.from_func(events)
def unenroll(events, experiment):
    return agg_any(all_([
        events.event_category == 'normandy',
        events.event_method == 'unenroll',
        events.event_string_value == experiment.experiment_slug,
    ]))


@Metric.from_func(telemetry_shield_study_parquet)
def unenroll_addon_expt(tssp):
    return agg_any(tssp.payload.data.study_state == 'exit')


@Metric.from_func(search_clients_daily)
def organic_search_count(scd):
    return agg_sum(scd.organic)


@Metric.from_func(events)
def view_about_logins(events):
    return agg_any(all_([
        events.event_method == 'open_management',
        events.event_category == 'pwmgr',
    ]))


@Metric.from_func(events)
def view_about_protections(events):
    return agg_any(all_([
        events.event_object == 'protection_report',
        events.event_method == 'show',
    ]))


@Metric.from_func(main_summary)
def fx_page_load_ms_2_mean(ms):
    return pings_histogram_mean(F.collect_list(ms.histogram_parent_fx_page_load_ms_2))


@Metric.from_func(main_summary)
def fx_tab_switch_total_e10s_ms_mean(ms):
    return pings_histogram_mean(F.collect_list(ms.histogram_parent_fx_tab_switch_total_e10s_ms))


@Metric.from_func(main_summary)
def about_home_topsites_first_paint(ms):
    return mean_narm(
        F.collect_list(
            F.when(ms.scalar_parent_timestamps_about_home_topsites_first_paint >= 0,
                   ms.scalar_parent_timestamps_about_home_topsites_first_paint)
        )
    )


@Metric.from_func(main_summary)
def first_paint(ms):
    return mean_narm(
        F.collect_list(
            F.when(ms.first_paint >= 0,
                   ms.first_paint)
        )
    )


