# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from mozanalysis.utils import all_


# An easy way to filter for particular event types
metric_library = {
    'session_start': {
        'category': 'action',
        'method': 'foreground',
        'object': 'app',
    },
    'session_end': {
        'category': 'action',
        'method': 'background',
        'object': 'app',
    },
    'erase_button': {
        'category': 'action',
        'method': 'click',
        'object': 'erase_button',
    },
    'back_button': {
        'category': 'action',
        'method': 'click',
        'object': 'back_button',
    },
    'notification': {
        'category': 'action',
        'method': 'click',
        'object': 'notification',
    },
    'notification_erase_open': {
        'category': 'action',
        'method': 'click',
        'object': 'notification_action',
        'value': 'erase_open',
    },
    'home_shortcut': {
        'category': 'action',
        'method': 'click',
        'object': 'notification_action',
        'value': 'erase_open',
    },
    'tabs_tray_erase': {
        'category': 'action',
        'method': 'click',
        'object': 'tabs_tray',
        'value': 'erase',
    },
    'recent_apps_remove': {
        'category': 'action',
        'method': 'click',
        'object': 'recent_apps',
        'value': 'erase',
    }
}


def make_select_col(event, metric_key):
    """Return an int Column named `metric_key`, given an event Column.

    In the returned Column, a row is 1 if the event is `metric_key`,
    otherwise it is 0.

    Useful when trying to count the number of occurrences of an event -
    just sum this column.

    Example usage:

        from pyspark.sql import functions as F

        t = spark.table('telemetry_mobile_event_parquet')
        t2 = t.filter(
            t.submission_date_s3 == '20190101'
        ).select(
            F.explode(t.events).alias('event')
        )
        t3 = t2.select(
            make_select_col(t2.event, 'session_start'),
            make_select_col(t2.event, 'session_end')
        )
        t3.agg(F.sum(t3.session_start), F.sum(t3.session_end)).collect()
    """
    return all_(
        event[k] == v for (k, v) in metric_library[metric_key].items()
    ).astype('int').alias(metric_key)


def make_where(event, metric_key):
    """Return a bool Column named `metric_key`, given an event Column.

    In the returned Column, a row is True iff the event is `metric_key`.

    Useful when filtering for an event.

    Example usage:

        t = spark.table('telemetry_mobile_event_parquet')
        t2 = t.filter(
            t.submission_date_s3 == '20190101'
        ).select(
            F.explode(t.events).alias('event')
        )
        t3 = t2.filter(make_where(t2.event, 'session_start'))
    """
    return all_(
        event[k] == v for (k, v) in metric_library[metric_key].items()
    ).alias(metric_key)
