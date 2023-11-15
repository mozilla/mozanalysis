# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.005, 0.025, 0.5, 0.975, 0.995)


def compare_samples(
    samples,
    ref_branch_label,
    individual_summary_quantiles=DEFAULT_QUANTILES,
    comparative_summary_quantiles=DEFAULT_QUANTILES,
):
    """Return descriptive statistics for branch stats and uplifts.

    Given per-branch samples for some quantity, return summary
    statistics (percentiles and the mean) for the quantity for each
    individual branch. Also return comparative summary statistics for
    the uplift of this quantity for each branch with respect to the
    reference branch.

    Args:
        samples (dict of pandas.Series or pandas.DataFrame): Each key
            is the label for a branch. Each value is the corresponding
            sample set.
        ref_branch_label (str): Label for the reference branch
            (typically the control).
        individual_summary_quantiles (list of float): Quantiles that
            define the summary stats for the individual branches'
            samples.
        comparative_summary_quantiles (list of float): Quantiles that
            define the summary stats for the comparative stats.

    Returns a dictionary:
        When the values of ``samples`` are Series, then this function
        returns a dictionary with the following keys and
        values:

            'individual': dictionary mapping each branch name to a
                pandas Series that holds the per-branch sample means and
                quantiles.
            'comparative': dictionary mapping each branch name to a
                pandas Series of summary statistics for the possible
                uplifts of the sampled quantity relative to the
                reference branch.

        Otherwise, when the values of ``samples`` are DataFrames, then
        this function returns a similar dictionary, except the Series
        are replaced with DataFrames. The index for each DataFrame is
        the columns of a value of ``samples``.
    """
    branch_list = list(samples.keys())

    return {
        "individual": {
            b: summarize_one_branch_samples(
                samples[b], quantiles=individual_summary_quantiles
            )
            for b in branch_list
        },
        "comparative": {
            b: summarize_joint_samples(
                samples[b],
                samples[ref_branch_label],
                quantiles=comparative_summary_quantiles,
            )
            for b in set(branch_list) - {ref_branch_label}
        },
    }


def summarize_one_branch_samples(samples, quantiles=DEFAULT_QUANTILES):
    """Return descriptive statistics for sampled population-level stats.

    Given samples from one or more distributions, calculate some
    quantiles and the mean.

    The intended primary use-case is for calculating credible intervals
    for stats when bootstrapping, or credible intervals around Bayesian
    model parameters; in both cases ``samples`` are from a posterior
    concerning one branch of an experiment.

    Args:
        samples (pandas.Series or pandas.DataFrame): Samples over which
            to compute the mean and quantiles.
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        If ``samples`` is a Series, then returns a pandas Series;
        the index contains the stringified ``quantiles`` plus
        ``'mean'``.

        If ``samples`` is a DataFrame, then returns a pandas DataFrame;
        the columns contain the stringified ``quantiles`` plus
        ``'mean'``. The index matches the columns of ``samples``.
    """
    if isinstance(samples, pd.DataFrame) or not np.isscalar(samples[0]):
        return _summarize_one_branch_samples_batch(samples, quantiles)
    else:
        return _summarize_one_branch_samples_single(samples, quantiles)


def summarize_joint_samples(focus, reference, quantiles=DEFAULT_QUANTILES):
    """Return descriptive statistics for uplifts.

    The intended use case of this function is to compare a 'focus'
    experiment branch to a 'reference' experiment branch (e.g. the
    control). Samples from each branch are combined pairwise; these
    pairs are considered to be samples from the joint probability
    distribution (JPD). We compute various quantities from the JPD:

    * We compute summary statistics for the distribution over relative
      uplifts ``focus / reference - 1``
    * We compute summary statistics for the distribution over absolute
      uplifts ``focus - reference``
    * We compute a summary statistic for the distribution over the L1
      norm of absolute uplifts ``abs(focus - reference)``
    * We compute the fraction of probability mass in the region
      ``focus > reference``, which in a Bayesian context may be
      interpreted as the probability that the ground truth model
      parameter is larger for the focus branch than the reference
      branch.

    ``focus`` and ``reference`` are samples from distributions; each is the
    same format that would be supplied to `summarize_one_branch_samples`
    when analyzing the branches independently.

    Can be used to analyse a single metric (supply Series as arguments)
    or in batch mode (supply DataFrames as arguments).

    Args:
        focus (pandas.Series or pandas.DataFrame): Bootstrapped samples
            or samples of a model parameter for a branch of an
            experiment. If a DataFrame, each column represents a
            different quantity.
        reference (pandas.Series or pandas.DataFrame): The same
            quantity, calculated for a different branch (typically the
            control).
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        A pandas Series or DataFrame containing a MultiIndex with the
        following labels on the higher level and stringified floats
        on the inner level

        * rel_uplift: Expectation value and quantiles over the relative
          uplift.
        * abs_uplift: Expectation value and quantiles over the absolute
          uplift.
        * max_abs_diff: Quantile 0.95 on the L1 norm of differences/
          absolute uplifts. In a Bayesian context, there is a 95%
          probability that the absolute difference is less than this in
          either direction.
        * prob_win: In a Bayesian context, the probability that the ground
          truth model parameter is larger for the focus than the reference
          branch.

        If returning a DataFrame, this MultiIndex is for the columns, and
        the index matches the columns of ``focus``.
    """
    if isinstance(focus, pd.DataFrame) or not np.isscalar(focus[0]):
        return _summarize_joint_samples_batch(focus, reference, quantiles)
    else:
        return _summarize_joint_samples_single(focus, reference, quantiles)


def _summarize_one_branch_samples_single(samples, quantiles=DEFAULT_QUANTILES):
    if not isinstance(samples, (pd.Series, np.ndarray, list)):
        # Hey pd.Series.agg - don't apply me elementwise!
        # Raising this error allows ``_summarize_one_branch_samples_batch``
        # to work also for non-batch ``samples`` (i.e. doing double duty)
        raise TypeError("Can't summarize a scalar")

    q_index = [str(v) for v in quantiles]

    res = pd.Series(index=q_index + ["mean"], dtype="float")

    res[q_index] = np.quantile(samples, quantiles)
    res["mean"] = np.mean(samples)
    return res


def _summarize_one_branch_samples_batch(samples, quantiles=DEFAULT_QUANTILES):
    return samples.agg(summarize_one_branch_samples, quantiles=quantiles).T


def _summarize_joint_samples_single(focus, reference, quantiles=DEFAULT_QUANTILES):
    str_quantiles = [str(q) for q in quantiles]

    index = pd.MultiIndex.from_tuples(
        [("rel_uplift", q) for q in str_quantiles + ["exp", "p_value"]]
        + [("abs_uplift", q) for q in str_quantiles + ["exp", "p_value"]]
        + [("max_abs_diff", "0.95"), ("prob_win",)]
    )

    res = pd.Series(index=index, dtype="float")

    rel_uplift_samples = focus / reference - 1
    res.loc[[("rel_uplift", q) for q in str_quantiles]] = np.quantile(
        rel_uplift_samples, quantiles
    )

    res.loc[("rel_uplift", "exp")] = np.mean(rel_uplift_samples)
    res.loc[("rel_uplift", "p_value")] = _bootstrap_p_value(rel_uplift_samples)

    abs_uplift_samples = focus - reference
    res.loc[[("abs_uplift", q) for q in str_quantiles]] = np.quantile(
        abs_uplift_samples, quantiles
    )

    res.loc[("abs_uplift", "exp")] = np.mean(abs_uplift_samples)
    res.loc[("abs_uplift", "p_value")] = _bootstrap_p_value(abs_uplift_samples)

    res.loc[("max_abs_diff", "0.95")] = np.quantile(np.abs(abs_uplift_samples), 0.95)

    res.loc["prob_win"] = np.mean(focus > reference)

    return res


def _bootstrap_p_value(samples):
    """
    Computes a 2-tailed p-value for test of:
        * H_0: theta = 0 vs
        * H_a: theta != 0
    Leverages the duality between confidence intervals and p-values to "invert" the
    the confidence interval and extra a p-value. Follows the implementation here:
    https://www.modernstatisticswithr.com/modchapter.html#intervalinversion except
    uses a binary search to improve performance
    """
    precision = 1 / len(samples)
    alphas = np.arange(0, 1, precision)

    # check for degenerate case where all bootstrap samples fall on one side of zero
    # if that's the case, return the smallest p_value we have precision to detect
    if min(samples) > 0 or max(samples) < 0:
        return alphas[1]

    low, high = 0, len(alphas)
    while low <= high:
        if high == 0:
            return alphas[1]
        mid = (low + high) // 2
        candidate_alpha = alphas[mid]

        cmp = _bootstrap_p_value_check_candidate_alpha(
            samples, candidate_alpha, precision
        )
        if cmp == 0:
            return candidate_alpha
        elif cmp < 0:
            low = mid + 1
        elif cmp > 0:
            high = mid - 1

    raise ValueError("p_value not found")


def _bootstrap_p_value_check_candidate_alpha(samples, alpha: float, precision: float):
    """
    Checks a candidate alpha to determine if this alpha is the transition
    point. That is, if CIs from smaller alphas contain zero whereas CIs from
    this alpha (or larger) do not contain zero.
    """
    this_interval = np.quantile(samples, [alpha / 2, 1 - (alpha / 2)])
    prev_interval = np.quantile(
        samples, [(alpha - precision) / 2, 1 - ((alpha - precision) / 2)]
    )
    this_interval_does_not_contain_zero = this_interval[0] > 0 or this_interval[1] < 0
    prev_interval_does_not_contain_zero = prev_interval[0] > 0 or prev_interval[1] < 0
    if this_interval_does_not_contain_zero and prev_interval_does_not_contain_zero:
        return 1  # too high
    elif (not this_interval_does_not_contain_zero) and (
        not prev_interval_does_not_contain_zero
    ):
        return -1  # too low
    elif this_interval_does_not_contain_zero and (
        not prev_interval_does_not_contain_zero
    ):
        return 0
    else:
        raise ValueError()


def _summarize_joint_samples_batch(focus, reference, quantiles=DEFAULT_QUANTILES):
    if set(focus.columns) != set(reference.columns):
        raise ValueError()

    return pd.DataFrame(
        {
            k: summarize_joint_samples(focus[k], reference[k], quantiles)
            for k in focus.columns
        },
        columns=focus.columns,
    ).T
