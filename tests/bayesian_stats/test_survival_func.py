# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import pytest
import scipy.stats as st

import mozanalysis.bayesian_stats.survival_func as mabssf


def test_get_thresholds():
    # One nonzero data point per percentile: 1 ... 101
    # Reverse order to make life slightly more interesting
    data = pd.Series(range(102)[::-1])
    res = mabssf.get_thresholds(data)

    assert len(data) - 1 == 101
    assert len(res) == 101

    for d, r in zip(data.sort_values()[:-1], res, strict=False):
        assert r == pytest.approx(d)


def test_get_thresholds_2():
    data = pd.Series(index=range(1000)).fillna(0)
    data[30:40] = 1
    data[874] = 2
    data[300] = 4
    data[0] = 9001

    res = mabssf.get_thresholds(data)

    assert set(res) == {0, 1, 2, 4}


def test_one_thresh():
    df = pd.DataFrame(columns=["branch", "val"], index=range(1000))
    df.iloc[::2, 0] = "control"
    df.iloc[1::2, 0] = "test"
    df.fillna(0, inplace=True)
    df.iloc[:300, 1] = range(300)

    # Odd threshold: both branches have equal amounts of data above/below
    # Specifically, 129 above, 371 below
    res = mabssf._one_thresh(41, df, "val", "control")

    assert res["individual"]["test"].loc["0.025"] == pytest.approx(
        st.beta(129 + 1, 371 + 1).ppf(0.025), abs=1e-6
    )
    assert res["individual"]["test"].loc["0.5"] == pytest.approx(
        st.beta(129 + 1, 371 + 1).ppf(0.5), abs=1e-6
    )
    assert (
        res["individual"]["control"].loc["0.025"]
        == res["individual"]["test"].loc["0.025"]
    )
    assert (
        res["individual"]["control"].loc["0.5"] == res["individual"]["test"].loc["0.5"]
    )

    assert res["individual"]["test"].loc["0.5"] == pytest.approx(
        1.0 * (df.val > 41).sum() / len(df.val), abs=0.001
    )

    assert "_tmp_threshold_val" not in df.columns

    # Even threshold: 'control' has one fewer data point above it
    res2 = mabssf._one_thresh(42, df, "val", "control")
    assert res2["individual"]["test"].loc["0.025"] == pytest.approx(
        st.beta(129 + 1, 371 + 1).ppf(0.025), abs=1e-6
    )
    assert res2["individual"]["test"].loc["0.5"] == pytest.approx(
        st.beta(129 + 1, 371 + 1).ppf(0.5), abs=1e-6
    )
    assert (
        res2["individual"]["control"].loc["0.025"]
        < res2["individual"]["test"].loc["0.025"]
    )
    assert (
        res2["individual"]["control"].loc["0.5"] < res2["individual"]["test"].loc["0.5"]
    )

    assert "_tmp_threshold_val" not in df.columns


def test_compare_branches():
    df = pd.DataFrame(columns=["branch", "val"], index=range(1000), dtype="float")
    df.iloc[::2, 0] = "control"
    df.iloc[1::2, 0] = "test"
    df.iloc[:300, 1] = range(300)

    with pytest.raises(ValueError, match="'df' contains null values for 'val'"):
        mabssf.compare_branches(df, "val", thresholds=[0.0, 15.9])

    df.iloc[300:, 1] = 0
    res = mabssf.compare_branches(df, "val", thresholds=[0.0, 15.9])

    assert res["individual"]["control"].loc[0.0, "0.025"] == pytest.approx(
        st.beta(149 + 1, 351 + 1).ppf(0.025), abs=1e-6
    )
    assert res["individual"]["control"].loc[0.0, "0.5"] == pytest.approx(
        0.298537, abs=1e-6
    )

    assert res["individual"]["test"].loc[15.9, "0.025"] == pytest.approx(
        st.beta(142 + 1, 358 + 1).ppf(0.025), abs=1e-6
    )
    assert res["individual"]["test"].loc[15.9, "0.5"] == pytest.approx(
        0.284575, abs=1e-6
    )


def test_few_auto_thresholds():
    df = pd.DataFrame(columns=["branch", "val"], index=range(1000))
    df.iloc[::2, 0] = "control"
    df.iloc[1::2, 0] = "test"
    df.fillna(0, inplace=True)
    df.iloc[20:30] = 1
    df.iloc[405] = 100

    res = mabssf.compare_branches(df, "val")
    assert set(res["individual"]["test"].index) == {0, 1}
    assert set(res["individual"]["control"].index) == {0, 1}
