# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import scipy.stats as st

from mozanalysis.stats.summarize_samples import (
    default_quantiles, compare_two_sample_sets
)


def compare(df, col_label, ref_branch_label='control', num_samples=10000):
    """Jointly sample conversion rates for branches then compare them.

    See `compare_from_agg` for more details.

    Args:
        df: a pandas DataFrame of queried experiment data in the standard
            format.
        col_label: Label for the df column contaning the metric to be
            analyzed.
        ref_branch_label: String in `df['branch']` that identifies the
            the branch with respect to which we want to calculate
            uplifts - usually the control branch.
        num_samples: The number of samples to compute.

    Returns a dictionary:
        'comparative': dictionary mapping branch names to a pandas
            Series of summary statistics for the possible uplifts of the
            conversion rate relative to the reference branch - see docs
            for `compare_two_sample_sets`.
        'individual': dictionary mapping branch names to a pandas
            Series of summary stats for the posterior distribution over
            the branch's conversion rate.
    """
    agg_col = aggregate_col(df, col_label)

    return compare_from_agg(
        agg_col, ref_branch_label=ref_branch_label, num_samples=num_samples
    )


def compare_many(df, col_label, num_samples=10000):
    """Jointly sample conversion rates for many branches then compare them.

    See `compare_many_from_agg` for more details.

    Args:
        df: a pandas DataFrame of experiment data. Each row represents
            data about an individual test subject. One column is named
            'branch' and contains the test subject's branch. The other
            columns contain the test subject's values for each metric.
            The column to be analyzed (named `col_label`) should be
            boolean or 0s and 1s.
        col_label: Label for the df column contaning the metric to be
            analyzed.
        num_samples: The number of samples to compute

    Returns a pandas.DataFrame of summary statistics for the possible
    uplifts:
        - columns: equivalent to rows output by `compare_two()`
        - index: list of branches
    """
    agg_col = aggregate_col(df, col_label)

    return compare_many_from_agg(
        agg_col, num_samples=num_samples
    )


def aggregate_col(df, col_label):
    # I would have used `isin` but it seems to be ~100x slower?
    if not ((df[col_label] == 0) | (df[col_label] == 1)).all():
        raise ValueError("All values in column '{}' must be 0 or 1.".format(col_label))

    return df[col_label].groupby('branch').agg({
        'num_enrollments': len,
        'num_conversions': np.sum
    })


def summarize_one_from_agg(
    s, num_enrollments_label='num_enrollments', num_conversions_label='num_conversions',
    quantiles=default_quantiles
):
    res = pd.Series()
    res['mean'] = s.loc[num_conversions_label] / s.loc[num_enrollments_label]

    ppfs = quantiles
    res[[str(v) for v in ppfs]] = st.beta(
        s.loc[num_conversions_label] + 1,
        s.loc[num_enrollments_label] - s.loc[num_conversions_label] + 1
    ).ppf(ppfs)

    return res


def compare_from_agg(
    df,
    ref_branch_label='control',
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
            - df.index lists the experiment branches
            - df.columns is
                `[num_enrollments_label, num_conversions_label]`
        ref_branch_label: Label for the df row containing data for the
            control branch
        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a dictionary:
        'comparative': dictionary mapping branch names to a pandas
            Series of summary statistics for the possible uplifts of the
            conversion rate relative to the reference branch - see docs
            for `compare_two_sample_sets`.
        'individual': dictionary mapping branch names to a pandas
            Series of summary stats for the posterior distribution over
            the branch's conversion rate.
    """
    assert ref_branch_label in df.index, "What's the reference branch?"

    samples = _generate_samples(
        df, num_enrollments_label, num_conversions_label, num_samples
    )

    # TODO: should 'comparative' and 'individual' be dfs?
    return {
        'comparative': {
                b: compare_two_sample_sets(
                    samples[b], samples[ref_branch_label]
                ) for b in df.index.drop(ref_branch_label)
            },
        'individual': {
                b: summarize_one_from_agg(
                    df.loc[b], num_enrollments_label, num_conversions_label
                ) for b in df.index
            },
    }


def compare_many_from_agg(
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

    comparative = pd.DataFrame(index=df.index, columns=res_columns)
    comparative.name = num_conversions_label

    individual = {}

    for branch in df.index:
        # Compare this branch to the best of the rest
        # (beware Monty's Revenge!)
        this_branch = samples[branch]
        # Warning: assumes we're trying to maximise the metric
        best_of_rest = samples.drop(branch, axis='columns').max(axis='columns')

        comparative.loc[branch] = compare_two_sample_sets(this_branch, best_of_rest)
        individual[branch] = summarize_one_from_agg(
            df.loc[branch], num_enrollments_label, num_conversions_label
        )

    return {
        'comparative': comparative,
        'individual': individual
    }


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
