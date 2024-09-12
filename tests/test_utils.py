# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

from mozanalysis.utils import add_days, all_, any_, date_sub, filter_outliers


def test_all_():
    df = pd.DataFrame(
        data=[
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [True, False, True],
        ],
        columns=["all_true", "all_false", "mixed"],
    )
    pres = pd.DataFrame(
        {
            "true_1": all_([df.all_true, df.all_true, df.all_true]),
            "false_2": all_([df.all_true, df.all_false]),
            "false_3": all_([df.all_false, df.all_false]),
            "false_4": all_([df.mixed, df.all_false]),
            "mixed_5": all_([df.mixed, df.all_true]),
        }
    )

    assert pres.shape == (4, 5)
    assert not pres.isnull().any().any()
    assert pres["true_1"].all()
    assert not pres["false_2"].any()
    assert not pres["false_3"].any()
    assert not pres["false_4"].any()
    assert not pres["mixed_5"][::2].any()
    assert pres["mixed_5"][1::2].all()

    # Check this workaround is still necessary:
    with pytest.raises(ValueError, match=r"The truth value of a Series is ambiguous.*"):
        all([df.all_true, df.all_true])


def test_any_():
    df = pd.DataFrame(
        data=[
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [True, False, True],
        ],
        columns=["all_true", "all_false", "mixed"],
    )
    pres = pd.DataFrame(
        {
            "true_1": any_([df.all_true, df.all_true, df.all_true]),
            "true_2": any_([df.all_true, df.all_false]),
            "false_3": any_([df.all_false, df.all_false]),
            "true_4": any_([df.mixed, df.all_true]),
            "mixed_5": any_([df.mixed, df.all_false]),
        }
    )
    assert pres.shape == (4, 5)
    assert not pres.isnull().any().any()
    assert pres["true_1"].all()
    assert pres["true_2"].any()
    assert not pres["false_3"].any()
    assert pres["true_4"].any()
    assert not pres["mixed_5"][::2].any()
    assert pres["mixed_5"][1::2].all()

    # Check this workaround is still necessary:
    with pytest.raises(ValueError, match=r"The truth value of a Series is ambiguous.*"):
        any([df.all_true, df.all_true])


def test_add_days():
    assert add_days("2019-01-01", 0) == "2019-01-01"
    assert add_days("2019-01-01", 1) == "2019-01-02"
    assert add_days("2019-01-01", -1) == "2018-12-31"


def test_date_sub():
    assert date_sub("2019-01-01", "2019-01-01") == 0
    assert date_sub("2019-01-02", "2019-01-01") == 1
    assert date_sub("2019-01-01", "2019-01-02") == -1


def test_filter_outliers():
    data = np.arange(100) + 1

    filtered = filter_outliers(data, 0.99)
    assert len(filtered) == 100
    assert filtered.max() == 99.01
    assert data.max() == 100


def test_filter_outliers_2():
    data = np.ones(100)

    filtered = filter_outliers(data, 0.99)
    assert len(filtered) == 100
