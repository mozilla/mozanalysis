# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.bayesian_stats as mabs


def test_summarize_one_branch_samples():
    s = pd.Series(np.linspace(0, 1, 1001))

    res = mabs.summarize_one_branch_samples(s, [0.05, 0.31, 0.95])
    assert res.shape == (4,)
    assert res['0.05'] == pytest.approx(0.05)
    assert res['0.31'] == pytest.approx(0.31)
    assert res['0.95'] == pytest.approx(0.95)
    assert res['mean'] == pytest.approx(0.5)


def test_summarize_one_branch_samples_batch():
    s = pd.Series(np.linspace(0, 1, 1001))
    df = pd.DataFrame({'a': s, 'b': s + 1})
    res = mabs.summarize_one_branch_samples(df, quantiles=[0.05, 0.31, 0.95])
    assert res.shape == (2, 4)

    assert res.loc['a', '0.05'] == pytest.approx(0.05)
    assert res.loc['a', '0.31'] == pytest.approx(0.31)
    assert res.loc['a', '0.95'] == pytest.approx(0.95)
    assert res.loc['a', 'mean'] == pytest.approx(0.5)

    assert res.loc['b', '0.05'] == pytest.approx(1.05)
    assert res.loc['b', '0.31'] == pytest.approx(1.31)
    assert res.loc['b', '0.95'] == pytest.approx(1.95)
    assert res.loc['b', 'mean'] == pytest.approx(1.5)


def test_summarize_joint_samples_trivial():
    quantiles = (0.05, 0.31, 0.95)
    res = mabs.summarize_joint_samples(
        pd.Series([6, 6, 6]), pd.Series([3, 3, 3]), quantiles=quantiles
    )
    assert res[('rel_uplift', 'exp')] == 1.
    assert res[('abs_uplift', 'exp')] == 3.
    assert res[('prob_win', None)] == 1
    assert res[('max_abs_diff', '0.95')] == 3.

    assert res[('rel_uplift', '0.05')] == 1.
    assert res[('rel_uplift', '0.31')] == 1.
    assert res[('rel_uplift', '0.95')] == 1.

    assert res[('abs_uplift', '0.05')] == 3.
    assert res[('abs_uplift', '0.31')] == 3.
    assert res[('abs_uplift', '0.95')] == 3.


def test_summarize_joint_samples_batch_trivial():
    quantiles = (0.05, 0.31, 0.95)
    res = mabs.summarize_joint_samples(
        pd.DataFrame({'a': [6, 6, 6], 'b': [1, 1, 1]}, columns=['a', 'b']),
        pd.DataFrame({'a': [3, 3, 3], 'b': [1, 1, 1]}, columns=['b', 'a']),
        quantiles=quantiles
    )
    assert res.loc['a', ('rel_uplift', 'exp')] == 1.
    assert res.loc['a', ('abs_uplift', 'exp')] == 3.
    assert res.loc['a', ('prob_win', None)] == 1
    assert res.loc['a', ('max_abs_diff', '0.95')] == 3.

    assert res.loc['a', ('rel_uplift', '0.05')] == 1.
    assert res.loc['a', ('rel_uplift', '0.31')] == 1.
    assert res.loc['a', ('rel_uplift', '0.95')] == 1.

    assert res.loc['a', ('abs_uplift', '0.05')] == 3.
    assert res.loc['a', ('abs_uplift', '0.31')] == 3.
    assert res.loc['a', ('abs_uplift', '0.95')] == 3.

    assert res.loc['b', ('rel_uplift', 'exp')] == 0.
    assert res.loc['b', ('abs_uplift', 'exp')] == 0.
    assert res.loc['b', ('max_abs_diff', '0.95')] == 0.

    assert res.loc['b', ('rel_uplift', '0.05')] == 0.
    assert res.loc['b', ('rel_uplift', '0.31')] == 0.
    assert res.loc['b', ('rel_uplift', '0.95')] == 0.

    assert res.loc['b', ('abs_uplift', '0.05')] == 0.
    assert res.loc['b', ('abs_uplift', '0.31')] == 0.
    assert res.loc['b', ('abs_uplift', '0.95')] == 0.
