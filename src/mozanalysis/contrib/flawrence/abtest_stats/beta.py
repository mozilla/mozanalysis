# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np

from mozanalysis.contrib.flawrence.abtest_stats import (
    res_columns, compare_two_sample_sets
)


def compare_two(
    df,
    control_label='control',
    num_enrollments_label='num_enrollments',
    num_conversions_label='num_conversions',
    num_samples=10000
):
    """Jointly sample conversion rates for two branches then compare them.

    Calculates various quantiles on the uplift of the non-control
    branch's sampled conversion rates with respect to the control
    branch's sampled conversion rates.

    The data in `df` is modelled as being generated binomially, with a
    Beta(1, 1) (uniform) prior over the conversion rate parameter.

    Args:
        df: A pandas dataframe of integers:
            - df.index lists the two experiment branches
            - df.columns is
                (num_enrollments_label, num_conversions_label)
        control_label: Label for the df row containing data for the
            control branch
        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a pandas.Series of summary statistics for the possible
    uplifts - see docs for `compare_two_sample_sets`
    """
    assert len(df.index) == 2
    assert control_label in df.index, "Which branch is the control?"

    test_label = list(set(df.index) - {control_label})[0]

    samples = _generate_samples(
        df, num_enrollments_label, num_conversions_label, num_samples
    )

    res = compare_two_sample_sets(samples[test_label], samples[control_label])
    res.name = num_conversions_label

    return res


def compare_many(
    df,
    num_enrollments_label='num_enrollments',
    num_conversions_label='num_conversions',
    num_samples=10000
):
    """Jointly sample conversion rates for many branches then compare them.

    Calculates various quantiles on the uplift of each branch's sampled
    conversion rates, with respect to the best of the other branches'
    sampled conversion rates.

    The data in `df` is modelled as being generated binomially, with a
    Beta(1, 1) (uniform) prior over the conversion rate parameter.

    Args:
        df: A pandas dataframe of integers:
            - df.index lists the experiment branches
            - df.columns is
                (num_enrollments_label, num_conversions_label)
        control_label: Label for the df row containing data for the
            control branch
        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a pandas.DataFrame of summary statistics for the possible
    uplifts:
        - columns: equivalent to rows output by `compare_two()`
        - index: list of branches
    """
    samples = _generate_samples(
        df, num_enrollments_label, num_conversions_label, num_samples
    )

    res = pd.DataFrame(index=df.index, columns=res_columns)
    res.name = num_conversions_label

    for branch in df.index:
        # Compare this branch to the best of the rest
        # (beware Monty's Revenge!)
        this_branch = samples[branch]
        # Warning: assumes we're trying to maximise the metric
        best_of_rest = samples.drop(branch, axis='columns').max(axis='columns')

        res.loc[branch] = compare_two_sample_sets(this_branch, best_of_rest)

    return res


def _generate_samples(
    df, num_enrollments_label, num_conversions_label, num_samples
):
    """Return samples from Beta distributions.

    Assumes a Beta(1, 1) prior.

        Args:
        df: A pandas dataframe of integers:
            - df.index lists the experiment branches
            - df.columns is
                (num_enrollments_label, num_conversions_label)
        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a pandas.DataFrame of sampled conversion rates:
        - columns: list of branches
        - index: enumeration of samples
    """
    samples = pd.DataFrame(index=np.arange(num_samples), columns=df.index)
    for branch_label, r in df.iterrows():
        # Oh, for a prior...
        samples[branch_label] = np.random.beta(
            r.loc[num_conversions_label] + 1,
            r.loc[num_enrollments_label] - r.loc[num_conversions_label] + 1,
            size=num_samples
        )

    return samples
