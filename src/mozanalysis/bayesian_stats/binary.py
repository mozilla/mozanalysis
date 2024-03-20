# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import scipy.stats as st

import mozanalysis.bayesian_stats as mabs


def compare_branches(
    df,
    col_label,
    ref_branch_label="control",
    num_samples=10000,
    individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles=mabs.DEFAULT_QUANTILES,
):
    """Jointly sample conversion rates for branches then compare them.

    See `compare_branches_from_agg` for more details.

    Args:
        df (pd.DataFrame): Queried experiment data in the standard
            format.
        col_label (str): Label for the df column contaning the metric
            to be analyzed.
        ref_branch_label (str, optional): String in ``df['branch']``
            that identifies the the branch with respect to which we
            want to calculate uplifts - usually the control branch.
        num_samples (int, optional): The number of samples to compute.
        individual_summary_quantiles (list, optional): Quantiles to
            determine the confidence bands on individual branch
            statistics. Change these when making Bonferroni corrections.
        comparative_summary_quantiles (list, optional): Quantiles to
            determine the confidence bands on comparative branch
            statistics (i.e. the change relative to the reference
            branch, probably the control). Change these when making
            Bonferroni corrections.

    Returns a dictionary:

        * 'individual': dictionary mapping branch names to a pandas
          Series of summary stats for the posterior distribution over
          the branch's conversion rate.
        * 'comparative': dictionary mapping branch names to a pandas
          Series of summary statistics for the possible uplifts of the
          conversion rate relative to the reference branch - see docs
          for
          :meth:`mozanalysis.bayesian_stats.summarize_samples.summarize_joint_samples`.
    """
    agg_col = aggregate_col(df, col_label)

    return compare_branches_from_agg(
        agg_col,
        ref_branch_label=ref_branch_label,
        num_samples=num_samples,
        individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
        comparative_summary_quantiles=mabs.DEFAULT_QUANTILES,
    )


def aggregate_col(df, col_label):
    """Return the number of enrollments and conversions per branch.

    Args:
        df (pd.DataFrame): Queried experiment data in the standard
            format.
        col_label (str): Label for the df column contaning the metric
            to be analyzed.

    Returns:
        A DataFrame. The index is the list of branches. It has the
        following columns:

        * num_enrollments: The number of experiment subjects enrolled in
          this branch who were eligible for the metric.
        * num_conversions: The number of these enrolled experiment subjects
          who met the metric's conversion criteria.
    """
    # I would have used `isin` but it seems to be ~100x slower?
    if not ((df[col_label] == 0) | (df[col_label] == 1)).all():
        raise ValueError(f"All values in column '{col_label}' must be 0 or 1.")

    return (
        df.groupby("branch")[col_label]
        .agg(["count", "sum"])
        .rename(columns={"count": "num_enrollments", "sum": "num_conversions"})
    )


def summarize_one_branch_from_agg(
    s,
    num_enrollments_label="num_enrollments",
    num_conversions_label="num_conversions",
    quantiles=mabs.DEFAULT_QUANTILES,
):
    """Return stats about a branch's conversion rate.

    Calculate and return a Series of summary stats for the posterior
    distribution over the branch's conversion rate.

    Args:
        s (pd.Series): Holds the number of enrollments and number of
            conversions for this branch and metric.
        num_enrollments_label (str, optional): The label in this Series
            for the number of enrollments
        num_conversions_label (str, optional): The label in this Series
            for the number of conversions
        quantiles (list, optional): The quantiles to return as summary
            statistics.

    Returns:
        A pandas Series; the index contains the stringified
        ``quantiles`` plus ``'mean'``.
    """
    beta = st.beta(
        s.loc[num_conversions_label] + 1,
        s.loc[num_enrollments_label] - s.loc[num_conversions_label] + 1,
    )

    q_index = [str(v) for v in quantiles]

    res = pd.Series(index=q_index + ["mean"], dtype=float)

    res[q_index] = beta.ppf(quantiles)
    res["mean"] = beta.mean()

    return res


def compare_branches_from_agg(
    df,
    ref_branch_label="control",
    num_enrollments_label="num_enrollments",
    num_conversions_label="num_conversions",
    num_samples=10000,
    individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles=mabs.DEFAULT_QUANTILES,
):
    """Jointly sample conversion rates for two branches then compare them.

    Calculates various quantiles on the uplift of the non-control
    branch's sampled conversion rates with respect to the control
    branch's sampled conversion rates.

    The data in `df` is modelled as being generated binomially, with a
    Beta(1, 1) (uniform) prior over the conversion rate parameter.

    Args:
        df: A pandas dataframe of integers.

            * ``df.index`` lists the experiment branches
            * ``df.columns`` is
              ``[num_enrollments_label, num_conversions_label]``

        ref_branch_label (str, optional): Label for the df row
            containing data for the control branch
        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a dictionary:

        * 'individual': dictionary mapping branch names to a pandas
          Series of summary stats for the posterior distribution over
          the branch's conversion rate.
        * 'comparative': dictionary mapping branch names to a pandas
          Series of summary statistics for the possible uplifts of the
          conversion rate relative to the reference branch - see docs
          for
          :meth:`mozanalysis.stats.summarize_samples.summarize_joint_samples`.
    """
    assert ref_branch_label in df.index, "What's the reference branch?"

    samples = get_samples(df, num_enrollments_label, num_conversions_label, num_samples)

    return {
        "individual": {
            b: summarize_one_branch_from_agg(
                df.loc[b],
                num_enrollments_label,
                num_conversions_label,
                quantiles=individual_summary_quantiles,
            )
            for b in df.index
        },
        "comparative": {
            b: mabs.summarize_joint_samples(
                samples[b],
                samples[ref_branch_label],
                quantiles=comparative_summary_quantiles,
            )
            for b in df.index.drop(ref_branch_label)
        },
    }


def get_samples(df, num_enrollments_label, num_conversions_label, num_samples):
    """Return samples from Beta distributions.

    Assumes a Beta(1, 1) prior.

    Args:
        df: A pandas dataframe of integers:

            * ``df.index`` lists the experiment branches
            * ``df.columns`` is
              ``(num_enrollments_label, num_conversions_label)``

        num_enrollments_label: Label for the df column containing the
            number of enrollments in each branch.
        num_conversions_label: Label for the df column containing the
            number of conversions in each branch.
        num_samples: The number of samples to compute

    Returns a pandas.DataFrame of sampled conversion rates

        * columns: list of branches
        * index: enumeration of samples
    """
    samples = pd.DataFrame(index=np.arange(num_samples), columns=df.index)
    for branch_label, r in df.iterrows():
        # Oh, for a better prior...
        samples[branch_label] = np.random.beta(
            r.loc[num_conversions_label] + 1,
            r.loc[num_enrollments_label] - r.loc[num_conversions_label] + 1,
            size=num_samples,
        )

    return samples
