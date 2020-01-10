# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import datetime
import hashlib
import numpy as np
from functools import reduce


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
    original_date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    new_date = original_date + datetime.timedelta(days=n_days)
    return datetime.datetime.strftime(new_date, '%Y-%m-%d')


def date_sub(date_string_l, date_string_r):
    """Return the number of days between two date strings like '2019-01-01'"""
    date_l = datetime.datetime.strptime(date_string_l, '%Y-%m-%d')
    date_r = datetime.datetime.strptime(date_string_r, '%Y-%m-%d')
    return (date_l - date_r).days


def filter_outliers(branch_data, threshold_quantile):
    """Return branch_data with outliers removed.

    N.B. `branch_data` is for an individual branch: if you do it for
    the entire experiment population in whole, then you may bias the
    results.

    TODO: here we remove outliers - should we have an option or
    default to cap them instead?

    Args:
        branch_data: Data for one branch as a 1D ndarray or similar.
        threshold_quantile (float): Discard outliers above this
            quantile.

    Returns:
        The subset of branch_data that was at or below the threshold
        quantile.
    """
    if threshold_quantile >= 1 or threshold_quantile < 0.5:
        raise ValueError("'threshold_quantile' should be close to 1")

    threshold_val = np.quantile(branch_data, threshold_quantile)

    return branch_data[branch_data <= threshold_val]


def hash_ish(string, hex_chars=12):
    """Return a crude hash of a string."""
    return hashlib.sha256(string.encode('utf-8')).hexdigest()[:hex_chars]
