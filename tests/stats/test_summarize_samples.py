# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.stats.summarize_samples as masss


def test_summarize_one_sample_set():
    s = pd.Series(np.linspace(0, 1, 1001))

    res = masss.summarize_one_sample_set(s, [0.05, 0.31, 0.95])
    assert res['0.05'] == pytest.approx(0.05)
    assert res['0.31'] == pytest.approx(0.31)
    assert res['0.95'] == pytest.approx(0.95)
    assert res['mean'] == pytest.approx(0.5)


def test_compare_two_sample_sets_trivial():
    quantiles = (0.05, 0.31, 0.95)
    res = masss.compare_two_sample_sets(
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
