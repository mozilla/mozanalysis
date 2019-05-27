# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.stats.summarize_samples as masss


def test_summarize_one_branch_samples():
    s = pd.Series(np.linspace(0, 1, 1001))

    res = masss.summarize_one_branch_samples(s, [0.05, 0.31, 0.95])
    assert res.shape == (4,)
    assert res['0.05'] == pytest.approx(0.05)
    assert res['0.31'] == pytest.approx(0.31)
    assert res['0.95'] == pytest.approx(0.95)
    assert res['mean'] == pytest.approx(0.5)


def test_summarize_one_branch_samples_batch():
    s = pd.Series(np.linspace(0, 1, 1001))
    df = pd.DataFrame({'a': s, 'b': s + 1})
    res = df.agg(masss.summarize_one_branch_samples_batch, quantiles=[0.05, 0.31, 0.95])
    assert res.shape == (4, 2)

    assert res.loc['0.05', 'a'] == pytest.approx(0.05)
    assert res.loc['0.31', 'a'] == pytest.approx(0.31)
    assert res.loc['0.95', 'a'] == pytest.approx(0.95)
    assert res.loc['mean', 'a'] == pytest.approx(0.5)

    assert res.loc['0.05', 'b'] == pytest.approx(1.05)
    assert res.loc['0.31', 'b'] == pytest.approx(1.31)
    assert res.loc['0.95', 'b'] == pytest.approx(1.95)
    assert res.loc['mean', 'b'] == pytest.approx(1.5)


def test_summarize_joint_samples_trivial():
    quantiles = (0.05, 0.31, 0.95)
    res = masss.summarize_joint_samples(
        pd.Series([6, 6, 6]), pd.Series([3, 3, 3]), quantiles=quantiles
    )
    assert res['rel_uplift_exp'] == 1.
    assert res['abs_uplift_exp'] == 3.
    assert res['prob_win'] == 1
    assert res['max_abs_diff_0.95'] == 3.

    assert res['rel_uplift_0.05'] == 1.
    assert res['rel_uplift_0.31'] == 1.
    assert res['rel_uplift_0.95'] == 1.

    assert res['abs_uplift_0.05'] == 3.
    assert res['abs_uplift_0.31'] == 3.
    assert res['abs_uplift_0.95'] == 3.


def test_summarize_joint_samples_batch_trivial():
    quantiles = (0.05, 0.31, 0.95)
    res = masss.summarize_joint_samples_batch(
        pd.DataFrame({'a': [6, 6, 6], 'b': [1, 1, 1]}, columns=['a', 'b']),
        pd.DataFrame({'a': [3, 3, 3], 'b': [1, 1, 1]}, columns=['b', 'a']),
        quantiles=quantiles
    )
    assert res.loc['a', 'rel_uplift_exp'] == 1.
    assert res.loc['a', 'abs_uplift_exp'] == 3.
    assert res.loc['a', 'prob_win'] == 1
    assert res.loc['a', 'max_abs_diff_0.95'] == 3.

    assert res.loc['a', 'rel_uplift_0.05'] == 1.
    assert res.loc['a', 'rel_uplift_0.31'] == 1.
    assert res.loc['a', 'rel_uplift_0.95'] == 1.

    assert res.loc['a', 'abs_uplift_0.05'] == 3.
    assert res.loc['a', 'abs_uplift_0.31'] == 3.
    assert res.loc['a', 'abs_uplift_0.95'] == 3.

    assert res.loc['b', 'rel_uplift_exp'] == 0.
    assert res.loc['b', 'abs_uplift_exp'] == 0.
    assert res.loc['b', 'max_abs_diff_0.95'] == 0.

    assert res.loc['b', 'rel_uplift_0.05'] == 0.
    assert res.loc['b', 'rel_uplift_0.31'] == 0.
    assert res.loc['b', 'rel_uplift_0.95'] == 0.

    assert res.loc['b', 'abs_uplift_0.05'] == 0.
    assert res.loc['b', 'abs_uplift_0.31'] == 0.
    assert res.loc['b', 'abs_uplift_0.95'] == 0.