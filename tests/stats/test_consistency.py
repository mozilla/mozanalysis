# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

import mozanalysis.stats.bayesian_binary as masbb
import mozanalysis.stats.bootstrap as masb


def test_bootstrap_vs_beta(spark_context):
    num_enrollments = 10000
    fake_data = pd.Series(np.zeros(num_enrollments))
    fake_data[:300] = 1

    boot_res = masb.bootstrap_one_branch(spark_context, fake_data)
    beta_res = masbb.summarize_one_branch_from_agg(pd.Series({
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
