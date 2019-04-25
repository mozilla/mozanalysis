# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pyspark.sql.functions as F
import pytest

from mozanalysis.utils import dedupe_columns, all_, any_, add_days


def test_dedupe_columns(spark):
    # NOTE: We are limited in the amount of testing we can do here b/c things
    # like `sorted([F.col(...)])` and checking for equality fail b/c of the
    # overridden magic methods on `Column` types.
    assert len(dedupe_columns(["a", "b", "a"])) == 2
    assert len(dedupe_columns(["a", "b", F.col("a")])) == 2
    assert len(dedupe_columns([F.col("a"), F.col("b"), F.col("a")])) == 2
    assert len(dedupe_columns([F.expr("a"), F.col("b"), F.col("a")])) == 2


def test_all_(spark):
    df = spark.createDataFrame(
        data=[
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [True, False, True],
        ],
        schema=["all_true", "all_false", "mixed"],
    )
    pres = df.select(
        all_([df.all_true, df.all_true, df.all_true]).alias("true_1"),
        all_([df.all_true, df.all_false]).alias("false_2"),
        all_([df.all_false, df.all_false]).alias("false_3"),
        all_([df.mixed, df.all_false]).alias("false_4"),
        all_([df.mixed, df.all_true]).alias("mixed_5"),
    ).toPandas()
    assert pres.shape == (4, 5)
    assert not pres.isnull().any().any()
    assert pres["true_1"].all()
    assert not pres["false_2"].any()
    assert not pres["false_3"].any()
    assert not pres["false_4"].any()
    assert not pres["mixed_5"][::2].any()
    assert pres["mixed_5"][1::2].all()

    # Check this workaround is still necessary:
    with pytest.raises(ValueError):
        all([df.all_true, df.all_true])


def test_any_(spark):
    df = spark.createDataFrame(
        data=[
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [True, False, True],
        ],
        schema=["all_true", "all_false", "mixed"],
    )
    pres = df.select(
        any_([df.all_true, df.all_true, df.all_true]).alias("true_1"),
        any_([df.all_true, df.all_false]).alias("true_2"),
        any_([df.all_false, df.all_false]).alias("false_3"),
        any_([df.mixed, df.all_true]).alias("true_4"),
        any_([df.mixed, df.all_false]).alias("mixed_5"),
    ).toPandas()
    assert pres.shape == (4, 5)
    assert not pres.isnull().any().any()
    assert pres["true_1"].all()
    assert pres["true_2"].any()
    assert not pres["false_3"].any()
    assert pres["true_4"].any()
    assert not pres["mixed_5"][::2].any()
    assert pres["mixed_5"][1::2].all()

    # Check this workaround is still necessary:
    with pytest.raises(ValueError):
        any([df.all_true, df.all_true])


def test_add_days():
    assert add_days('20190101', 0) == '20190101'
    assert add_days('20190101', 1) == '20190102'
    assert add_days('20190101', -1) == '20181231'
