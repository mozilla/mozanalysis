# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from scipy.stats import norm
from math import pi
from statsmodels.stats.power import tt_ind_solve_power


def mann_whitney_solve_sample_size(
    rel_effect_size,
    alpha=.05,
    power=.8,
    parent_distribution="normal"
):

    """
    Wilcoxen-Mann-Whitney rank sum test sample size calculation,
    based on asymptotic efficiency relative to the t-test.
    """

    are = {
        "uniform": 1.,
        "normal": 3./pi,
        "logistic": (pi**2)/9.,
        "laplace": 1.5
    }

    if parent_distribution not in are.keys():
        raise ValueError(f"Parent distribution must be in {are.keys()}")

    t_sample_size = tt_ind_solve_power(
        effect_size=rel_effect_size,
        power=power,
        alpha=alpha
    )
    return t_sample_size*are[parent_distribution]


def poisson_diff_solve_sample_size(
    mean,
    effect_size,
    alpha=.05,
    power=.90
):

    z_alpha = norm.ppf(1-alpha/2)
    z_power = norm.ppf(power)

    denom = (effect_size/(z_alpha+z_power))**2
    return (mean+effect_size)/denom
