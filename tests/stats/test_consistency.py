# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.stats.bayesian_binary as masbbeta
import mozanalysis.stats.bayesian_bootstrap as masbboot
import mozanalysis.stats.bootstrap as masb
import mozanalysis.stats.summarize_samples as masss


def percentile_bootstrap(spark_context, data):
    samples, _ = masb.get_bootstrap_samples(spark_context, data)

    return masss.summarize_one_branch_samples(samples)


def test_bootstrap_vs_beta(spark_context):
    num_enrollments = 10000
    fake_data = pd.Series(np.zeros(num_enrollments))
    fake_data[:300] = 1

    boot_res = masb.bootstrap_one_branch(spark_context, fake_data)
    beta_res = masbbeta.summarize_one_branch_from_agg(pd.Series({
        # `-1` to simulate Beta(0, 0) improper prior, closer to
        # bootstrap for quantiles (i think?)
        'num_enrollments': len(fake_data) - 1,
        'num_conversions': fake_data.sum() - 1
    }))

    for l in boot_res.index:
        # Bootstrapped quantiles are discretized based on the number of enrollments,
        # which sets `abs`.
        #
        # Is `num_samples` large enough to consistently achieve results that
        # match the beta model to within the accuracy of this discrete limit?
        # Not quite. So we backed it off a bit and ask for the bootstrapped result
        # to be within 1.9 quanta of the beta result - which was enough for a
        # percentile bootstrap
        #
        # Though the empirical bootstrap seems
        assert boot_res.loc[l] == pytest.approx(
            beta_res.loc[l],
            # 1.9 is good enough with a percentile bootstrap
            # abs=1.9/num_enrollments
            # Empirical bootstrap requires a wider tolerance band (?!)
            abs=4.9/num_enrollments
        ), l


def test_percentile_bootstrap_vs_beta(spark_context):
    num_enrollments = 10000
    fake_data = pd.Series(np.zeros(num_enrollments))
    fake_data[:300] = 1

    boot_res = percentile_bootstrap(spark_context, fake_data)
    beta_res = masbbeta.summarize_one_branch_from_agg(pd.Series({
        # `-1` to simulate Beta(0, 0) improper prior, closer to
        # bootstrap for quantiles (i think?)
        'num_enrollments': len(fake_data) - 1,
        'num_conversions': fake_data.sum() - 1
    }))

    for l in boot_res.index:
        # Bootstrapped quantiles are discretized based on the number of enrollments,
        # which sets `abs`.
        #
        # Is `num_samples` large enough to consistently achieve results that
        # match the beta model to within the accuracy of this discrete limit?
        # Not quite. So we backed it off a bit and ask for the bootstrapped result
        # to be within 1.9 quanta of the beta result - which was enough for a
        # percentile bootstrap
        assert boot_res.loc[l] == pytest.approx(
            beta_res.loc[l],
            # abs=1.9 is usually good enough with a percentile bootstrap
            # set abs=2.9 because there are lots of tests
            abs=2.9/num_enrollments
        ), l


def test_bayesian_bootstrap_vs_beta(spark_context):
    # The two distributions should be mathematically identical for binary data
    # like this; differences could emerge from
    # 1. implementation errors
    # 2. not taking enough bootstrap replications to suppress variance
    num_enrollments = 10000
    fake_data = pd.Series(np.zeros(num_enrollments))
    fake_data[:300] = 1

    boot_res = masbboot.bootstrap_one_branch(spark_context, fake_data)
    beta_res = masbbeta.summarize_one_branch_from_agg(pd.Series({
        # `-1` to simulate Beta(0, 0) improper prior, closer to
        # bootstrap for quantiles (i think?)
        'num_enrollments': len(fake_data) - 1,
        'num_conversions': fake_data.sum() - 1
    }))

    for l in boot_res.index:
        assert boot_res.loc[l] == pytest.approx(
            beta_res.loc[l],
            # 1.9 is good enough with a percentile bootstrap
            # abs=1.9/num_enrollments
            abs=1.9/num_enrollments
        ), (l, boot_res, beta_res)


def test_bayesian_bootstrap_vs_percentile_bootstrap_geometric(spark_context):
    num_enrollments = 20000

    rs = np.random.RandomState(42)
    data = rs.geometric(p=0.1, size=num_enrollments)

    bb_res = masbboot.bootstrap_one_branch(spark_context, data)
    pboot_res = percentile_bootstrap(spark_context, data)

    assert bb_res['mean'] == pytest.approx(10, rel=1e-2)
    assert bb_res['0.5'] == pytest.approx(10, rel=1e-2)

    for l in bb_res.index:
        assert bb_res.loc[l] == pytest.approx(
            pboot_res.loc[l],
            rel=5e-3
        ), (l, bb_res, pboot_res)


def test_bayesian_bootstrap_vs_percentile_bootstrap_poisson(spark_context):
    num_enrollments = 10001

    rs = np.random.RandomState(42)
    data = rs.poisson(lam=10, size=num_enrollments)

    bb_res = masbboot.bootstrap_one_branch(spark_context, data)
    pboot_res = percentile_bootstrap(spark_context, data)

    assert bb_res['mean'] == pytest.approx(10, rel=1e-2)
    assert bb_res['0.5'] == pytest.approx(10, rel=1e-2)

    for l in bb_res.index:
        assert bb_res.loc[l] == pytest.approx(
            pboot_res.loc[l],
            rel=5e-3
        ), (l, bb_res, pboot_res)
