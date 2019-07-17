# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import pytest

import mozanalysis.bayesian_stats.binary as mabsb
import mozanalysis.bayesian_stats as mabs


def test_aggregate_col():
    df = pd.DataFrame([
        ['ctl', 1, 0],
        ['ctl', 1, 0],
        ['ctl', 0, 1],
        ['tst', 1, 0],
        ['tst', 0, 1],
    ], columns=['branch', 'val1', 'val2'])

    res1 = mabsb.aggregate_col(df, 'val1')
    assert res1.loc['ctl', 'num_enrollments'] == 3
    assert res1.loc['tst', 'num_enrollments'] == 2
    assert res1.loc['ctl', 'num_conversions'] == 2
    assert res1.loc['tst', 'num_conversions'] == 1

    res2 = mabsb.aggregate_col(df, 'val2')
    assert res2.loc['ctl', 'num_enrollments'] == 3
    assert res2.loc['tst', 'num_enrollments'] == 2
    assert res2.loc['ctl', 'num_conversions'] == 1
    assert res2.loc['tst', 'num_conversions'] == 1


def test_summarize_one_branch_vs_samples():
    s = pd.Series([30, 80], index=['num_conversions', 'num_enrollments'])

    ppf = mabsb.summarize_one_branch_from_agg(s, quantiles=[0.5, 0.41])

    samples = mabsb.get_samples(
        s.to_frame().T, 'num_enrollments', 'num_conversions', 100000
    )

    res = mabs.summarize_one_branch_samples(samples.iloc[:, 0], quantiles=[0.5, 0.41])

    assert ppf['0.5'] == pytest.approx(res['0.5'], abs=0.001)
    assert ppf['0.41'] == pytest.approx(res['0.41'], abs=0.001)
    assert ppf['mean'] == pytest.approx(res['mean'], abs=0.001)


def test_compare_branches():
    # The stats has been checked above; this just exercises the remaining
    # business logic

    df = pd.DataFrame([
        ['ctl', 1, 0],
        ['ctl', 1, 0],
        ['ctl', 0, 1],
        ['tst', 1, 0],
        ['tst', 0, 1],
        ['3rd', 1, 0],
        ['3rd', 0, 1],
    ], columns=['branch', 'val1', 'val2'])

    res = mabsb.compare_branches(df, 'val2', ref_branch_label='ctl')

    assert set(res['individual'].keys()) == set(df.branch.unique())
    assert set(res['comparative'].keys()) == set(df.branch.unique()) - set(['ctl'])
