# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats as mabs
from mozanalysis.utils import filter_outliers

# Functions that return highly processed stats


def bootstrap_one_branch(
    sc, data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=mabs.DEFAULT_QUANTILES
):
    """Bootstrap on the means of one branch on its own.

    Generates ``num_samples`` sampled means, then returns summary
    statistics for their distribution.

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

    return summarize_one_branch_empirical_bootstrap(
        samples, original_sample, summary_quantiles
    )


# Functions that return per-resampled-population stats


# TODO: Think carefully about the naming of these functions w.r.t.
# the above functions; and maybe give the two levels of samples
# different names (i.e. it's maybe not clear that here 'sample'
# refers to a statistic calculated over a resampled population)


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


# Functions that calculate stats over one resampled population


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


# Bootstrap-specific stats


def summarize_one_branch_empirical_bootstrap(samples, original_sample_stats, quantiles):
    inv_quantiles = [1 - q for q in quantiles]

    if not isinstance(original_sample_stats, dict):
        # `stat_fn` returned a scalar: non-batch mode.
        assert isinstance(samples, pd.Series)

        res = original_sample_stats - mabs.summarize_one_branch_samples_single(
            samples - original_sample_stats, inv_quantiles
        )

        res.index = _update_index(res.index, quantiles, inv_quantiles)

    else:
        assert isinstance(samples, pd.DataFrame)
        osss = pd.Series(original_sample_stats)

        res = mabs.summarize_one_branch_samples_batch(
            samples.sub(osss), inv_quantiles
        ).rsub(osss, axis='rows')

        res.columns = _update_index(res.columns, quantiles, inv_quantiles)

    return res


def _update_index(index, quantiles, inv_quantiles):
    for i, iq in enumerate(inv_quantiles):
        assert index[i] == str(iq)

    return [str(q) for q in quantiles] + list(index[len(quantiles):])
