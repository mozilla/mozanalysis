# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb


def test_resample_and_agg_once():
    assert mabsbb._resample_and_agg_once(
        np.array([3.]), np.array([3]), mabsbb.bb_mean
    ) == pytest.approx(3.)


def test_resample_and_agg_once_multistat(stack_depth=0):
    data_values = np.array([0, 1])
    data_counts = np.array([10000, 10000])
    res = mabsbb._resample_and_agg_once(
        data_values,
        data_counts,
        lambda x, y: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.dot(x, y),
        },
    )

    assert res['min'] == 0
    assert res['max'] == 1
    assert res['mean'] == pytest.approx(0.5, rel=1e-1)

    if stack_depth >= 3:
        assert res['mean'] != 0.5  # Extremely unlikely
    elif res['mean'] == 0.5:
        # This is a 0.5% event - implausible but not impossible.
        # Re-roll the dice a few times to make sure this was a fluke.
        test_resample_and_agg_once_multistat(stack_depth + 1)


def test_resample_and_agg_once_bcast(spark_context):
    b_data_values = spark_context.broadcast(np.array([3.]))
    b_data_counts = spark_context.broadcast(np.array([3]))
    assert mabsbb._resample_and_agg_once_bcast(
        b_data_values, b_data_counts, mabsbb.bb_mean, 42
    ) == 3.


def test_get_bootstrap_samples(spark_context):
    res = mabsbb.get_bootstrap_samples(
        np.array([3., 3., 3.]), num_samples=2, sc=spark_context
    )
    assert res.shape == (2,)
    assert res[0] == pytest.approx(3.)
    assert res[1] == pytest.approx(3.)


def test_get_bootstrap_samples_no_spark():
    test_get_bootstrap_samples(None)


def test_get_bootstrap_samples_multistat(spark_context, stack_depth=0):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mabsbb.get_bootstrap_samples(
        data,
        lambda x, y: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.dot(x, y),
        },
        num_samples=2,
        sc=spark_context
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
        test_get_bootstrap_samples_multistat(spark_context, stack_depth + 1)


def test_get_bootstrap_samples_multistat_no_spark():
    test_get_bootstrap_samples_multistat(None)


def test_bootstrap_one_branch(spark_context):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mabsbb.bootstrap_one_branch(
        data, num_samples=100, summary_quantiles=(0.5, 0.61), sc=spark_context
    )

    assert res['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.61'] == pytest.approx(0.5, rel=1e-1)


def test_bootstrap_one_branch_no_spark():
    test_bootstrap_one_branch(None)


def test_bootstrap_one_branch_multistat(spark_context):
    data = np.concatenate([np.zeros(10000), np.ones(10000), [1e20]])
    res = mabsbb.bootstrap_one_branch(
        data,
        stat_fn=lambda x, y: {
            'max': np.max(x),
            'mean': np.dot(x, y),
        },
        num_samples=5,
        summary_quantiles=(0.5, 0.61),
        threshold_quantile=0.9999,
        sc=spark_context,
    )

    assert res.shape == (2, 3)

    assert res.loc['max', 'mean'] == 1
    assert res.loc['max', '0.5'] == 1
    assert res.loc['max', '0.61'] == 1
    assert res.loc['mean', 'mean'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.61'] == pytest.approx(0.5, rel=1e-1)


def test_bootstrap_one_branch_multistat_no_spark():
    test_bootstrap_one_branch_multistat(None)


def test_compare_branches(spark_context_or_none):
    data = pd.DataFrame(
        index=range(60000),
        columns=['branch', 'val'],
        dtype='float'
    )
    data.iloc[::3, 0] = 'control'
    data.iloc[1::3, 0] = 'same'
    data.iloc[2::3, 0] = 'bigger'

    data.iloc[::2, 1] = 0
    data.iloc[1::2, 1] = 1

    data.iloc[2::12, 1] = 1

    assert data.val[data.branch != 'bigger'].mean() == 0.5
    assert data.val[data.branch == 'bigger'].mean() == pytest.approx(0.75)

    res = mabsbb.compare_branches(data, 'val', num_samples=2, sc=spark_context_or_none)

    assert res['individual']['control']['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['same']['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['individual']['bigger']['mean'] == pytest.approx(0.75, rel=1e-1)

    assert 'control' not in res['comparative'].keys()
    assert res['comparative']['same'][('rel_uplift', 'exp')] == \
        pytest.approx(0, abs=0.1)
    assert res['comparative']['bigger'][('rel_uplift', 'exp')] == \
        pytest.approx(0.5, abs=0.1)

    # num_samples=2 so only 3 possible outcomes
    assert res['comparative']['same'][('prob_win', None)] in (0, 0.5, 1)
    assert res['comparative']['bigger'][('prob_win', None)] == \
        pytest.approx(1, abs=0.01)


def test_compare_branches_multistat(spark_context_or_none):
    data = pd.DataFrame(
        index=range(60000),
        columns=['branch', 'val'],
        dtype='float'
    )
    data.iloc[::3, 0] = 'control'
    data.iloc[1::3, 0] = 'same'
    data.iloc[2::3, 0] = 'bigger'

    data.iloc[::2, 1] = 0
    data.iloc[1::2, 1] = 1

    data.iloc[2::12, 1] = 1

    assert data.val[data.branch != 'bigger'].mean() == 0.5
    assert data.val[data.branch == 'bigger'].mean() == pytest.approx(0.75)

    res = mabsbb.compare_branches(
        data,
        'val',
        stat_fn=lambda x, y: {
            'max': np.max(x),
            'mean': np.dot(x, y),
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


def test_bb_mean():
    values = np.array([0, 1, 2])
    weights = np.array([0.1, 0.4, 0.5])

    assert mabsbb.bb_mean(values, weights) == pytest.approx(1.4)


def test_bb_quantile():
    values = np.array([0, 1, 2])
    weights = np.array([0.1, 0.4, 0.5])

    calc_median = mabsbb.make_bb_quantile_closure(0.5)

    assert calc_median(values, weights) == pytest.approx(1)

    calc_a_bunch = mabsbb.make_bb_quantile_closure([0, 0.1, 0.5, 1])

    res = calc_a_bunch(values, weights)
    assert res[0] == pytest.approx(0)
    assert res[0.1] == pytest.approx(0)
    assert res[0.5] == pytest.approx(1)
    assert res[1] == pytest.approx(2)
