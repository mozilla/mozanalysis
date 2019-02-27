# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from mozanalysis.stats import _resample

from mozanalysis.contrib.flawrence.abtest_stats import (
    summarise_one_sample_set, compare_two_sample_sets
)


# TODO: break mozanalysis naming conventions: rename `num_iterations` to `num_samples`?

def bootstrap_one(sc, data, num_iterations=10000, seed_start=0):
    """Bootstrap on the means of one variation on its own.

    Generates `num_iterations` sampled means, then returns summary
    statistics for their distribution.

    Args:
        sc: The spark context
        data: The data as a list, numpy array, or pandas series
        num_iterations: The number of bootstrap iterations to perform
        seed_start: An int with which to seed numpy's RNG. It must
            be unique to this set of calculations.
    """
    samples = _resample_parallel(sc, data, num_iterations, seed_start)
    return summarise_one_sample_set(samples)


def bootstrap_two(sc, focus, reference, num_iterations=10000, filter_outliers=None):
    """Jointly sample bootstrapped means from two distributions then compare them.

    Calculates various quantiles on the uplift of the focus branch's
    mean value with respect to the reference branch's mean value.

    Args:
        sc: The spark context
        focus: The data for the focal branch as a list, numpy array, or
            pandas series
        reference: The data for the reference (typically control) branch
        num_iterations: The number of bootstrap iterations to perform
        filter_outliers: An optional threshold quantile, above which to
            discard outliers.

    Returns a dictionary:
        - 'comparative': pandas.Series of summary statistics for the possible
            uplifts - see docs for `compare_two_sample_sets`
        - 'individual': list of summary stats for (focus, reference) means.
            Each set of summary stats is a pandas.Series
    """
    # TODO: don't supply focus and reference separately - be like `..beta`?

    # FIXME: should we be filtering or truncating outliers?
    if filter_outliers:
        focus = focus[focus <= np.quantile(focus, filter_outliers)]
        reference = reference[reference <= np.quantile(reference, filter_outliers)]

    focus_samples = _resample_parallel(sc, focus, num_iterations)
    reference_samples = _resample_parallel(sc, reference, num_iterations)

    return {
        'comparative':
            compare_two_sample_sets(focus_samples, reference_samples),
        'individual': [
                summarise_one_sample_set(focus_samples),
                summarise_one_sample_set(reference_samples)
            ],
    }


def _resample_parallel(sc, data, num_iterations, seed_start=None):
    """Return bootstrapped samples for the mean of `data`.

    Do the resampling in parallel over the cluster.

    Args:
        sc: The spark context
        data: The data as a list, numpy array, or pandas series
        num_iterations: The number of samples to return
        seed_start: A seed for the random number generator; this
            function will use seeds in the range
                [seed_start, seed_start + num_iterations)
            and these particular seeds must not be used elsewhere
            in this calculation. By default, use a random seed.

    Returns a numpy array of sampled means
    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every iteration has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_iterations)

    try:
        broadcast_data = sc.broadcast(data)

        summary_stat_samples = sc.parallelize(seed_range).map(
            lambda seed: _resample(
                iteration=seed % np.iinfo(np.uint32).max,
                stat_fn=np.mean,
                broadcast_data=broadcast_data,
            )
        ).collect()

        return np.array(summary_stat_samples)

    finally:
        broadcast_data.unpersist()


def _resample_local(data, num_iterations):
    """Equivalent to `_resample_parallel` but doesn't require Spark.

    The main purpose of this function is to document what's being done
    in `_resample_parallel` :D
    """
    return np.array([
        np.mean(np.random.choice(data, size=len(data)))
        for _ in range(num_iterations)
    ])
