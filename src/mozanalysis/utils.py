# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import datetime
import hashlib
from functools import reduce

import numpy as np


def all_(l):
    """Return the element-wise logical AND of `Column`s.

    Think of this as a vector-friendly version of the built-in function
    `all()`.

    Args:
        l: A list of `Column`s of booleans. Or, more generally,
            an iterable of items to reduce over.

    Returns:
        A `Column` of booleans representing the logical AND. Or,
            more generally, the result of the logical AND.
    """
    return reduce(lambda x, y: x & y, l)


def any_(l):
    """Return the element-wise logical OR of `Column`s.

    Think of this as a vector-friendly version of the built-in function
    `any()`.

    Args:
        l: A list of `Column`s of booleans. Or, more generally,
            an iterable of items to reduce over.

    Returns:
        A `Column` of booleans representing the logical OR. Or,
            more generally, the result of the logical OR.
    """
    return reduce(lambda x, y: x | y, l)


def add_days(date_string, n_days):
    """Add `n_days` days to a date string like '2019-01-01'."""
    original_date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    new_date = original_date + datetime.timedelta(days=n_days)
    return datetime.datetime.strftime(new_date, "%Y-%m-%d")


def date_sub(date_string_l, date_string_r):
    """Return the number of days between two date strings like '2019-01-01'"""
    date_l = datetime.datetime.strptime(date_string_l, "%Y-%m-%d")
    date_r = datetime.datetime.strptime(date_string_r, "%Y-%m-%d")
    return (date_l - date_r).days


def filter_outliers(branch_data, threshold_quantile):
    """Return branch_data with outliers capped.

    N.B. `branch_data` is for an individual branch: if you do it for
    the entire experiment population in whole, then you may bias the
    results.

    Args:
        branch_data: Data for one branch as a 1D ndarray or similar.
        threshold_quantile (float): Sets outliers above this
            quantile equal to the value of this quantile.

    Returns:
        branch_data with values capped at or below the threshold
        quantile.
    """
    if threshold_quantile > 1 or threshold_quantile < 0.5:
        raise ValueError("'threshold_quantile' should be close to, and <= 1")

    min_threshold = np.min(branch_data, axis=0)
    max_threshold = np.quantile(branch_data, threshold_quantile, axis=0)

    return np.clip(branch_data, min_threshold, max_threshold)


def hash_ish(string, hex_chars=12):
    """Return a crude hash of a string."""
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:hex_chars]


def get_time_intervals(
    start_date: str | datetime.datetime,
    interval: int,
    max_num_dates_enrollment: int,
) -> list[datetime.datetime]:
    """Use a start date and create end dates for enrollment intervals.
    Used to generate intervals of enrollment to calculate metrics over
    variable enrollment lengths.

    Args:
        start_date: First date of enrollment for sizing job.
        interval: Number of days to increment the enrollment end date by.
        max_num_dates_enrollment: Ceiling for the length of the enrollment
            period.

    Returns:
        date_list: List of dates where variable enrollment windows will
            end.
    """

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    date = start_date + datetime.timedelta(interval - 1)
    date_list = [date.date()]
    last_enrollment_date = start_date + datetime.timedelta(
        days=(max_num_dates_enrollment - 1)
    )
    while date < last_enrollment_date:
        date = date + datetime.timedelta(interval)
        date_list.append(date.date())

    date_list[len(date_list) - 1] = last_enrollment_date.date()
    return date_list
