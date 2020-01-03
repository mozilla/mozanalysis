# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.frequentist_stats.bootstrap as mafsb


def test_resample_and_agg_once():
    assert mafsb._resample_and_agg_once(np.array([3., 3., 3.]), np.mean) == 3.


def test_resample_and_agg_once_multistat(stack_depth=0):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mafsb._resample_and_agg_once(
        data,
        lambda x: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
        },
    )

    assert res['min'] == 0
    assert res['max'] == 1
    assert res['mean'] == pytest.approx(np.mean(data), rel=1e-1)

    if stack_depth >= 3:
        assert res['mean'] != np.mean(data)  # Extremely unlikely
    elif res['mean'] == np.mean(data):
        # This is a 0.5% event - implausible but not impossible.
        # Re-roll the dice a few times to make sure this was a fluke.
        test_resample_and_agg_once_multistat(stack_depth + 1)


def test_resample_and_agg_once_bcast(spark_context):
    b_data = spark_context.broadcast(np.array([3., 3., 3.]))
    assert mafsb._resample_and_agg_once_bcast(b_data, np.mean, 42) == 3.


def test_get_bootstrap_samples(spark_context_or_none):
    res = mafsb.get_bootstrap_samples(
        np.array([3., 3., 3.]), num_samples=2, sc=spark_context_or_none
    )
    assert res.shape == (2,)

    assert res[0] == 3.
    assert res[1] == 3.


def test_get_bootstrap_samples_multistat(spark_context_or_none, stack_depth=0):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mafsb.get_bootstrap_samples(
        data,
        lambda x: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=2,
        sc=spark_context_or_none,
    )

    assert res.shape == (2, 3)

    assert (res['min'] == 0).all()
    assert (res['max'] == 1).all()
    assert res['mean'].iloc[0] == pytest.approx(np.mean(data), rel=1e-1)
    assert res['mean'].iloc[1] == pytest.approx(np.mean(data), rel=1e-1)

    # If we stuff up (duplicate) the seeds then things aren't random
    assert res['mean'].iloc[0] != res['mean'].iloc[1]

    if stack_depth >= 3:
        assert (res['mean'] != np.mean(data)).any()  # Extremely unlikely
    elif (res['mean'] == np.mean(data)).any():
        # Re-roll the dice a few times to make sure this was a fluke.
        test_get_bootstrap_samples_multistat(spark_context_or_none, stack_depth + 1)


def test_bootstrap_one_branch(spark_context_or_none):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mafsb.bootstrap_one_branch(
        data, num_samples=100, summary_quantiles=(0.5, 0.61), sc=spark_context_or_none
    )

    assert res['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.61'] == pytest.approx(0.5, rel=1e-1)


def test_bootstrap_one_branch_multistat(spark_context_or_none):
    data = np.concatenate([np.zeros(10000), np.ones(10000), [1e20]])
    res = mafsb.bootstrap_one_branch(
        data,
        stat_fn=lambda x: {
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=5,
        summary_quantiles=(0.5, 0.61),
        threshold_quantile=0.9999,
        sc=spark_context_or_none
    )

    assert res.shape == (2, 3)

    assert res.loc['max', 'mean'] == 1
    assert res.loc['max', '0.5'] == 1
    assert res.loc['max', '0.61'] == 1
    assert res.loc['mean', 'mean'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.61'] == pytest.approx(0.5, rel=1e-1)


def test_compare_branches(spark_context_or_none):
    data = pd.DataFrame(
        index=range(60000),
        columns=['branch', 'val'],
        dtype='float',
    )
    data.iloc[::3, 0] = 'control'
    data.iloc[1::3, 0] = 'same'
    data.iloc[2::3, 0] = 'bigger'

    data.iloc[::2, 1] = 0
    data.iloc[1::2, 1] = 1

    data.iloc[2::12, 1] = 1

    assert data.val[data.branch != 'bigger'].mean() == 0.5
    assert data.val[data.branch == 'bigger'].mean() == pytest.approx(0.75)

    res = mafsb.compare_branches(data, 'val', num_samples=2, sc=spark_context_or_none)

    assert res['individual']['control']['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['same']['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['bigger']['mean'] == pytest.approx(0.75, rel=1e-1)

    assert 'control' not in res['comparative'].keys()
    assert res['comparative']['same'][('rel_uplift', 'exp')] == \
        pytest.approx(0, abs=0.1)
    assert res['comparative']['bigger'][('rel_uplift', 'exp')] == \
        pytest.approx(0.5, abs=0.1)

    # num_samples=2 so (0, 0.5, 1):
    assert res['comparative']['same'][('prob_win', None)] in (0, 0.5, 1)
    assert res['comparative']['bigger'][('prob_win', None)] == \
        pytest.approx(1, abs=0.01)


def test_compare_branches_multistat(spark_context_or_none):
    data = pd.DataFrame(
        index=range(60000),
        columns=['branch', 'val'],
        dtype='float',
    )
    data.iloc[::3, 0] = 'control'
    data.iloc[1::3, 0] = 'same'
    data.iloc[2::3, 0] = 'bigger'

    data.iloc[::2, 1] = 0
    data.iloc[1::2, 1] = 1

    data.iloc[2::12, 1] = 1

    assert data.val[data.branch != 'bigger'].mean() == 0.5
    assert data.val[data.branch == 'bigger'].mean() == pytest.approx(0.75)

    res = mafsb.compare_branches(
        data,
        'val',
        stat_fn=lambda x: {
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=2,
        sc=spark_context_or_none,
    )

    assert res['individual']['control'].loc['mean', 'mean'] \
        == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['same'].loc['mean', 'mean'] \
        == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['bigger'].loc['mean', 'mean'] \
        == pytest.approx(0.75, rel=1e-1)

    assert 'control' not in res['comparative'].keys()

    assert res['comparative']['same'].loc['mean', ('rel_uplift', 'exp')] \
        == pytest.approx(0, abs=0.1)
    assert res['comparative']['bigger'].loc['mean', ('rel_uplift', 'exp')] \
        == pytest.approx(0.5, abs=0.1)

    # num_samples=2 so only 3 possible outcomes
    assert res['comparative']['same'].loc['mean', ('prob_win', None)] in (0, 0.5, 1)
    assert res['comparative']['bigger'].loc['mean', ('prob_win', None)] \
        == pytest.approx(1, abs=0.01)

    assert res['comparative']['same'].loc['max', ('rel_uplift', 'exp')] == 0
    assert res['comparative']['bigger'].loc['max', ('rel_uplift', 'exp')] == 0
