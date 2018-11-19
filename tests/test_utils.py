# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pyspark.sql.functions as F

from mozanalysis.utils import dedupe_columns


def test_dedupe_columns(spark):
    # NOTE: We are limited in the amount of testing we can do here b/c things
    # like `sorted([F.col(...)])` and checking for equality fail b/c of the
    # overridden magic methods on `Column` types.
    assert len(dedupe_columns(["a", "b", "a"])) == 2
    assert len(dedupe_columns(["a", "b", F.col("a")])) == 2
    assert len(dedupe_columns([F.col("a"), F.col("b"), F.col("a")])) == 2
    assert len(dedupe_columns([F.expr("a"), F.col("b"), F.col("a")])) == 2
