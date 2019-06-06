# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import datetime
import numpy as np
from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql.column import Column


def dedupe_columns(columns):
    """
    Given a list of columns, returns a list with duplicates removed.

    These can be either from `F.col` or `F.expr`.

    NOTE: Because of the underlying way `Column`s are coded using Python
    magic methods, the usual methods for comparing or deduping don't work
    here. They aren't hashable so we can't use sets and they aren't
    comparable by normal methods. So we fall back to comparing the underlying
    Java object ids instead.

    """
    d = {}

    for col in columns:
        if not isinstance(col, Column):
            col = F.col(col)
        d[col._jc] = col

    return list(d.values())


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
    """Add `n_days` days to a date string like '20190101'."""
    original_date = datetime.datetime.strptime(date_string, '%Y%m%d')
    new_date = original_date + datetime.timedelta(days=n_days)
    return datetime.datetime.strftime(new_date, '%Y%m%d')


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
