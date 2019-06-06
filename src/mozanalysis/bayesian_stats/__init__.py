# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd


DEFAULT_QUANTILES = (0.005, 0.05, 0.5, 0.95, 0.995)


def summarize_one_branch_samples(samples, quantiles=DEFAULT_QUANTILES):
    if isinstance(samples, pd.DataFrame) or not np.isscalar(samples[0]):
        return summarize_one_branch_samples_batch(samples, quantiles)
    else:
        return summarize_one_branch_samples_single(samples, quantiles)


def summarize_joint_samples(focus, reference, quantiles=DEFAULT_QUANTILES):
    if isinstance(focus, pd.DataFrame) or not np.isscalar(focus[0]):
        return summarize_joint_samples_batch(focus, reference, quantiles)
    else:
        return summarize_joint_samples_single(focus, reference, quantiles)


def summarize_one_branch_samples_single(samples, quantiles=DEFAULT_QUANTILES):
    """Return descriptive statistics for sampled population-level stats.

    Given samples from a distribution, calculate some quantiles and the
    mean.

    The intended primary use-case is for calculating confidence
    intervals when bootstrapping, or probability bands around model
    parameters; in both cases ``samples`` relate to data from one branch
    of an experiment.

    Args:
        samples (pandas.Series): Samples over which to compute the mean
            and quantiles.
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        A pandas Series; the index contains the stringified
        ``quantiles`` plus ``'mean'``.
    """
    if not isinstance(samples, (pd.Series, np.ndarray, list)):
        # Hey pd.Series.agg - don't apply me elementwise!
        # Raising this error allows ``summarize_one_branch_samples_batch``
        # to work also for non-batch ``samples`` (i.e. doing double duty)
        raise TypeError("Can't summarize a scalar")

    q_index = [str(v) for v in quantiles]

    res = pd.Series(index=q_index + ['mean'])

    res[q_index] = np.quantile(samples, quantiles)
    res['mean'] = np.mean(samples)
    return res


def summarize_one_branch_samples_batch(samples, quantiles=DEFAULT_QUANTILES):
    """Return descriptive statistics for sampled population-level stats.

    Given samples from more than one distribution, calculate some
    quantiles and the means.

    The intended primary use-case is for calculating confidence
    intervals when bootstrapping, or probability bands around model
    parameters; in both cases ``samples`` relate to data from one branch
    of an experiment.

    Some example use-cases:

    * When building a time series for one metric, each column's
      samples are derived from data for a different analysis window.

    * When bootstrapping several quantiles simultaneously (with a
      ``stat_fn`` that returns several values), each column's samples
      come from bootstrapping a different quantile. N.B. in this
      example, the quantiles being bootstrapped over are unrelated to
      the ``quantiles`` argument of this function.

    Args:
        samples (pandas.DataFrame): Each column contains a set of
            samples over which to compute the means and quantiles.
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        A pandas DataFrame; the columns contain the stringified
        ``quantiles`` plus ``'mean'``. The index matches the columns of
        ``samples``.
    """
    return samples.agg(summarize_one_branch_samples, quantiles=quantiles).T


def summarize_joint_samples_single(focus, reference, quantiles=DEFAULT_QUANTILES):
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

    Args:
        focus (pandas.Series): Bootstrapped samples or samples of a
            model parameter for a branch of an experiment.
        reference (pandas.Series): The same quantity, calculated for a
            different branch (typically the control).
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        A pandas Series containing the following data

        * rel_uplift_*: Expectation value and quantiles over the relative
          uplift.
        * abs_uplift_*: Expectation value and quantiles over the absolute
          uplift.
        * max_abs_diff_0.95: Quantile 0.95 on the L1 norm of differences/
          absolute uplifts. In a Bayesian context, there is a 95%
          probability that the absolute difference is less than this in
          either direction.
        * prob_win: In a Bayesian context, the probability that the ground
          truth model parameter is larger for the focus than the reference
          branch.
    """
    rel_q_labels = ['rel_uplift_{}'.format(q) for q in quantiles]
    abs_q_labels = ['abs_uplift_{}'.format(q) for q in quantiles]

    res = pd.Series(index=rel_q_labels + ['rel_uplift_exp'] + abs_q_labels + [
        'abs_uplift_exp', 'max_abs_diff_0.95', 'prob_win'
    ])

    rel_uplift_samples = focus / reference - 1
    res[rel_q_labels] = np.quantile(rel_uplift_samples, quantiles)
    res['rel_uplift_exp'] = np.mean(rel_uplift_samples)

    abs_uplift_samples = focus - reference
    res[abs_q_labels] = np.quantile(abs_uplift_samples, quantiles)
    res['abs_uplift_exp'] = np.mean(abs_uplift_samples)

    res['max_abs_diff_0.95'] = np.quantile(np.abs(abs_uplift_samples), 0.95)

    res['prob_win'] = np.mean(focus > reference)

    return res


def summarize_joint_samples_batch(focus, reference, quantiles=DEFAULT_QUANTILES):
    """Batch version of `summarize_joint_samples`.

    See docs for `summarize_joint_samples`. The difference here is that
    ``focus`` and ``reference`` are DataFrames not Series; each column
    represents samples from a different distribution or statistic.

    Some example use-cases:

    * When building a time series for one metric, each column's
      samples are derived from data for a different analysis window.
    * When bootstrapping several quantiles simultaneously (with a
      ``stat_fn`` that returns several values), each column's samples
      come from bootstrapping a different quantile. N.B. in this
      example, the quantiles being bootstrapped over are unrelated to
      the ``quantiles`` argument of this function.

    Args:
        focus (pandas.DataFrame): Each column contains bootstrapped
            samples or samples of a model parameter for a branch of an
            experiment.
        reference (pandas.DataFrame): The same quantity, calculated for
            a different branch (typically the control).
        quantiles (list, optional): The quantiles to compute - a good
            reason to override the defaults would be when Bonferroni
            corrections are required.

    Returns:
        A pandas DataFrame with an index matching the columns of
        ``focus``, and the following columns:

        * rel_uplift_*: Expectation value and quantiles over the relative
          uplift.
        * abs_uplift_*: Expectation value and quantiles over the absolute
          uplift.
        * max_abs_diff_0.95: Quantile 0.95 on the L1 norm of differences/
          absolute uplifts. In a Bayesian context, there is a 95%
          probability that the absolute difference is less than this in
          either direction.
        * prob_win: In a Bayesian context, the probability that the ground
          truth model parameter is larger for the focus than the reference
          branch.
    """
    if set(focus.columns) != set(reference.columns):
        raise ValueError()

    return pd.DataFrame({
        k: summarize_joint_samples(focus[k], reference[k], quantiles)
        for k in focus.columns
    }, columns=focus.columns).T
