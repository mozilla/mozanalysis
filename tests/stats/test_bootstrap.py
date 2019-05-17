# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.stats.bootstrap as masb


def test_resample_and_agg_once():
    assert masb._resample_and_agg_once(np.array([3., 3., 3.]), np.mean) == 3.


def test_resample_and_agg_once_multistat():
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = masb._resample_and_agg_once(
        data,
        lambda x: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
        },
    )

    assert res['min'] == 0
    assert res['max'] == 1
    assert res['mean'] != np.mean(data)  # Extremely unlikely
    assert res['mean'] == pytest.approx(np.mean(data), rel=1e-1)


def test_resample_and_agg_once_bcast(spark_context):
    b_data = spark_context.broadcast(np.array([3., 3., 3.]))
    assert masb._resample_and_agg_once_bcast(b_data, np.mean, 42) == 3.


def test_get_bootstrap_samples(spark_context):
    res = masb.get_bootstrap_samples(
        spark_context, np.array([3., 3., 3.]), num_samples=2
    )
    assert res.shape == (2,)
    assert res[0] == 3.
    assert res[1] == 3.


def test_get_bootstrap_samples_multistat(spark_context):
    data = np.concatenate([np.zeros(10000), np.ones(10000)])
    res = masb.get_bootstrap_samples(
        spark_context,
        data,
        lambda x: {
            'min': np.min(x),
            'max': np.max(x),
            'mean': np.mean(x),
        },
        num_samples=2
    )
    print(res)

    assert res.shape == (2, 3)
    assert (res['min'] == 0).all()
    assert (res['max'] == 1).all()
    assert (res['mean'] != np.mean(data)).any()  # Extremely unlikely
    assert res['mean'].iloc[0] == pytest.approx(np.mean(data), rel=1e-1)
    assert res['mean'].iloc[1] == pytest.approx(np.mean(data), rel=1e-1)

    # If we stuff up (duplicate) the seeds then things aren't random
    assert res['mean'].iloc[0] != res['mean'].iloc[1]


def test_filter_outliers():
    data = np.arange(100) + 1

    filtered = masb._filter_outliers(data, 0.99)
    assert len(filtered) == 99
    assert filtered.max() == 99
    assert data.max() == 100


def test_filter_outliers_2():
    data = np.ones(100)

    filtered = masb._filter_outliers(data, 0.99)
    assert len(filtered) == 100
