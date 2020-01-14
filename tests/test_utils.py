# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pytest

from mozanalysis.utils import all_, any_, add_days, filter_outliers, \
    date_sub


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
    assert add_days('2019-01-01', 0) == '2019-01-01'
    assert add_days('2019-01-01', 1) == '2019-01-02'
    assert add_days('2019-01-01', -1) == '2018-12-31'


def test_date_sub():
    assert date_sub('2019-01-01', '2019-01-01') == 0
    assert date_sub('2019-01-02', '2019-01-01') == 1
    assert date_sub('2019-01-01', '2019-01-02') == -1


def test_filter_outliers():
    data = np.arange(100) + 1

    filtered = filter_outliers(data, 0.99)
    assert len(filtered) == 99
    assert filtered.max() == 99
    assert data.max() == 100


def test_filter_outliers_2():
    data = np.ones(100)

    filtered = filter_outliers(data, 0.99)
    assert len(filtered) == 100
