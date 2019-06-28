# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

from mozanalysis.bayesian_stats import DEFAULT_QUANTILES
from mozanalysis.utils import filter_outliers


def bootstrap_each_branch(
    sc, df, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=DEFAULT_QUANTILES
):
    """Run ``bootstrap_one_branch`` for each branch's data."""
    return {
        b: bootstrap_one_branch(
            sc, df[df.branch == b], stat_fn, num_samples, seed_start,
            threshold_quantile, summary_quantiles
        ) for b in df.branch.unique()
    }


def bootstrap_one_branch(
    sc, data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=DEFAULT_QUANTILES
):
    """Perform a basic bootstrap for one branch on its own.

    Resamples the data ``num_samples`` times, computes ``stat_fn`` for
    each sample, then returns summary statistics for the distribution
    of the outputs of ``stat_fn``.

    Args:
        sc: The spark context
        data: The data as a list, 1D numpy array, or pandas Series
        stat_fn: Either a function that aggregates each resampled
            population to a scalar (e.g. the default value ``np.mean``
            lets you bootstrap means), or a function that aggregates
            each resampled population to a dict of scalars. In both
            cases, this function must accept a one-dimensional ndarray
            as its input.
        num_samples: The number of bootstrap iterations to perform
        seed_start: An int with which to seed numpy's RNG. It must
            be unique within this set of calculations.
        threshold_quantile (float, optional): An optional threshold
            quantile, above which to discard outliers. E.g. ``0.9999``.
        summary_quantiles (list, optional): Quantiles to determine the
            confidence bands on the branch statistics. Change these
            when making Bonferroni corrections.
    """
    samples, original_sample = get_bootstrap_samples(
        sc, data, stat_fn, num_samples, seed_start, threshold_quantile
    )

    return summarize_one_branch_samples(
        samples, original_sample, summary_quantiles
    )


def summarize_one_branch_samples(samples, original_sample_stats, quantiles):
    """Return the mean with confidence intervals for bootstrap samples.

    Given samples for stats over bootstrap replicates, compute a basic
    bootstrap to estimate the mean and confidence intervals for it.

    Args:
        samples (pandas.Series or pandas.DataFrame): Samples over which
            to compute the mean and quantiles.
        original_sample_stats (float or pandas.Series): `stat_fn`
            computed over the original sample.
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
    if not isinstance(original_sample_stats, dict):
        # `stat_fn` returned a scalar: non-batch mode.
        assert isinstance(samples, pd.Series)

        return _summarize_one_branch_samples_single(
            samples, original_sample_stats, quantiles
        )

    else:
        assert isinstance(samples, pd.DataFrame)

        return _summarize_one_branch_samples_batch(
            samples, original_sample_stats, quantiles
        )


def _summarize_one_branch_samples_single(samples, original_sample_stats, quantiles):
    q_index = [str(v) for v in quantiles]
    res = pd.Series(index=q_index + ['mean'])

    inv_quantiles = [1 - q for q in quantiles]

    res[q_index] = 2 * original_sample_stats - np.quantile(samples, inv_quantiles)
    res['mean'] = 2 * original_sample_stats - np.mean(samples)
    return res


def _summarize_one_branch_samples_batch(samples, original_sample_stats, quantiles):
    return pd.DataFrame({
        k: _summarize_one_branch_samples_single(
            samples[k], original_sample_stats[k], quantiles
        )
        for k in original_sample_stats
    }).T


def get_bootstrap_samples(
    sc, data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None
):
    """Return ``stat_fn`` evaluated on resampled and original data.

    Do the resampling in parallel over the cluster.

    Args:
        sc: The spark context
        data: The data as a list, 1D numpy array, or pandas series
        stat_fn: Either a function that aggregates each resampled
            population to a scalar (e.g. the default value ``np.mean``
            lets you bootstrap means), or a function that aggregates
            each resampled population to a dict of scalars. In both
            cases, this function must accept a one-dimensional ndarray
            as its input.
        num_samples: The number of samples to return
        seed_start: A seed for the random number generator; this
            function will use seeds in the range::

                [seed_start, seed_start + num_samples)

            and these particular seeds must not be used elsewhere
            in this calculation. By default, use a random seed.
        threshold_quantile (float, optional): An optional threshold
            quantile, above which to discard outliers. E.g. ``0.9999``.

    Returns:
        A two-element tuple, with elements

            1. ``stat_fn`` evaluated over ``num_samples`` samples.

                * By default, a pandas Series of sampled means
                * if ``stat_fn`` returns a scalar, a pandas Series
                * if ``stat_fn`` returns a dict, a pandas DataFrame
                  with columns set to the dict keys.

            2. ``stat_fn`` evaluated on the (thresholded) original
               dataset.
    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    if np.isnan(data).any():
        raise ValueError("'data' contains null values")

    if threshold_quantile:
        data = filter_outliers(data, threshold_quantile)

    original_sample_stats = stat_fn(data)  # FIXME: should this be a series

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every call has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_samples)

    # TODO: run locally `if sc is None`?
    try:
        broadcast_data = sc.broadcast(data)

        summary_stat_samples = sc.parallelize(seed_range).map(
            lambda seed: _resample_and_agg_once_bcast(
                broadcast_data=broadcast_data,
                stat_fn=stat_fn,
                unique_seed=seed % np.iinfo(np.uint32).max,
            )
        ).collect()

    finally:
        broadcast_data.unpersist()

    summary_df = pd.DataFrame(summary_stat_samples)
    if len(summary_df.columns) == 1:
        # Return a Series if stat_fn returns a scalar
        return summary_df.iloc[:, 0], original_sample_stats

    # Else return a DataFrame if stat_fn returns a dict
    return summary_df, original_sample_stats


def _resample_and_agg_once_bcast(broadcast_data, stat_fn, unique_seed):
    return _resample_and_agg_once(
        broadcast_data.value, stat_fn, unique_seed
    )


def _resample_and_agg_once(data, stat_fn, unique_seed=None):
    random_state = np.random.RandomState(unique_seed)

    n = len(data)
    # TODO: can't we just use random_state.choice? Wouldn't that be faster?
    # There's not thaaat much difference in RAM requirements?
    randints = random_state.randint(0, n, n)
    resampled_data = data[randints]

    return stat_fn(resampled_data)
