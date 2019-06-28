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


def test_get_bootstrap_samples(spark_context):
    res = mafsb.get_bootstrap_samples(
        spark_context, np.array([3., 3., 3.]), num_samples=2
    )
    assert res[0].shape == (2,)

    assert res[1] == 3.

    assert res[0][0] == 3.
    assert res[0][1] == 3.


def test_get_bootstrap_samples_multistat(spark_context, stack_depth=0):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mafsb.get_bootstrap_samples(
        spark_context,
        data,
        lambda x: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=2
    )

    assert res[0].shape == (2, 3)

    assert res[1]['min'] == 0
    assert res[1]['max'] == 1
    assert res[1]['mean'] == pytest.approx(np.mean(data))

    assert (res[0]['min'] == 0).all()
    assert (res[0]['max'] == 1).all()
    assert res[0]['mean'].iloc[0] == pytest.approx(np.mean(data), rel=1e-1)
    assert res[0]['mean'].iloc[1] == pytest.approx(np.mean(data), rel=1e-1)

    # If we stuff up (duplicate) the seeds then things aren't random
    assert res[0]['mean'].iloc[0] != res[0]['mean'].iloc[1]

    if stack_depth >= 3:
        assert (res[0]['mean'] != np.mean(data)).any()  # Extremely unlikely
    elif (res[0]['mean'] == np.mean(data)).any():
        # Re-roll the dice a few times to make sure this was a fluke.
        test_get_bootstrap_samples_multistat(spark_context, stack_depth + 1)


def test_bootstrap_one_branch(spark_context):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = mafsb.bootstrap_one_branch(
        spark_context, data, num_samples=100, summary_quantiles=(0.5, 0.61)
    )

    assert res['mean'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res['0.61'] == pytest.approx(0.5, rel=1e-1)


def test_bootstrap_one_branch_multistat(spark_context):
    data = np.concatenate([np.zeros(10000), np.ones(10000), [1e20]])
    res = mafsb.bootstrap_one_branch(
        spark_context, data,
        stat_fn=lambda x: {
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=5,
        summary_quantiles=(0.5, 0.61),
        threshold_quantile=0.9999
    )

    assert res.shape == (2, 3)

    assert res.loc['max', 'mean'] == 1
    assert res.loc['max', '0.5'] == 1
    assert res.loc['max', '0.61'] == 1
    assert res.loc['mean', 'mean'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.5'] == pytest.approx(0.5, rel=1e-1)
    assert res.loc['mean', '0.61'] == pytest.approx(0.5, rel=1e-1)


def test_summarize_one_branch_samples():
    s = pd.Series(np.linspace(0, 1, 1001))

    res = mafsb.summarize_one_branch_samples(
        s, s.mean(), [0.05, 0.31, 0.95]
    )
    assert res.shape == (4,)
    assert res['0.05'] == pytest.approx(0.05)
    assert res['0.31'] == pytest.approx(0.31)
    assert res['0.95'] == pytest.approx(0.95)
    assert res['mean'] == pytest.approx(0.5)


def test_summarize_one_branch_samples_batch():
    s = pd.Series(np.linspace(0, 1, 1001))
    df = pd.DataFrame({'a': s, 'b': s + 1})
    res = mafsb.summarize_one_branch_samples(
        df, {'a': s.mean(), 'b': s.mean() + 1},
        quantiles=[0.05, 0.31, 0.95]
    )
    assert res.shape == (2, 4)

    assert res.loc['a', '0.05'] == pytest.approx(0.05)
    assert res.loc['a', '0.31'] == pytest.approx(0.31)
    assert res.loc['a', '0.95'] == pytest.approx(0.95)
    assert res.loc['a', 'mean'] == pytest.approx(0.5)

    assert res.loc['b', '0.05'] == pytest.approx(1.05)
    assert res.loc['b', '0.31'] == pytest.approx(1.31)
    assert res.loc['b', '0.95'] == pytest.approx(1.95)
    assert res.loc['b', 'mean'] == pytest.approx(1.5)
