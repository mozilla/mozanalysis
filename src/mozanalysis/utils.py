# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
