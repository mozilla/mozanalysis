# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

import mozanalysis.stats.summarize_samples as masss


# Functions that return highly processed stats


def compare_branches(
    sc, df, col_label, ref_branch_label='control', stat_fn=np.mean,
    num_samples=10000, threshold_quantile=None,
    individual_summary_quantiles=masss.default_quantiles,
    comparative_summary_quantiles=masss.default_quantiles
):
    """Jointly sample bootstrapped statistics then compare them.

    Args:
        sc: The Spark context
        df: a pandas DataFrame of queried experiment data in the
            standard format (see `mozanalysis.experiment`).
        col_label (str): Label for the df column contaning the metric
            to be analyzed.
        ref_branch_label (str, optional): String in `df['branch']` that
            identifies the branch with respect to which we want to
            calculate uplifts - usually the control branch.
        stat_fn (func, optional): A function that either:
            - Aggregates each resampled population to a scalar (e.g.
                the default, `np.mean`), or
            - Aggregates each resampled population to a dict of
                scalars.
            In both cases, this function must accept a one-dimensional
            ndarray or pandas Series as its input.
        num_samples (int, optional): The number of bootstrap iterations
            to perform.
        threshold_quantile (float, optional): An optional threshold
            quantile, above which to discard outliers. E.g. `0.9999`.
        individual_summary_quantiles (list, optional): Quantiles to
            determine the confidence bands on individual branch
            statistics. Change these when making Bonferroni corrections.
        comparative_summary_quantiles (list, optional): Quantiles to
            determine the confidence bands on comparative branch
            statistics (i.e. the change relative to the reference
            branch, probably the control). Change these when making
            Bonferroni corrections.

    Returns a dictionary:
    If `stat_fn` returns a scalar (this is the default), then
    this function returns a dictionary has the following keys and
    values:
        'individual': dictionary mapping each branch name to a pandas
            Series that holds the expected value for the bootstrapped
            `stat_fn`, and confidence intervals.
        'comparative': dictionary mapping each branch name to a pandas
            Series of summary statistics for the possible uplifts of
            the bootstrapped `stat_fn` relative to the reference branch.

    Otherwise, when `stat_fn` returns a dict, then this function
    returns a similar dictionary, except the Series are replaced with
    DataFrames. Each row in each DataFrame corresponds to one output
    of `stat_fn`, and is the Series that would be returned if `stat_fn`
    computed only this statistic.
    """
    branch_list = df.branch.unique()

    if ref_branch_label not in branch_list:
        raise ValueError("Branch label '{b}' not in branch list '{bl}".format(
            b=ref_branch_label, bl=branch_list
        ))

    samples = {
        # TODO: do we need to control seed_start? If so then we must be careful here
        b: get_bootstrap_samples(
            sc,
            df[col_label][df.branch == b],
            stat_fn,
            num_samples,
            threshold_quantile=threshold_quantile
        ) for b in branch_list
    }

    # Depending on whether `stat_fn` returns a scalar or a dict,
    # we might have to call the 'batch' versions:
    if isinstance(samples[branch_list[0]], pd.Series):
        summarize_one = masss.summarize_one_branch_samples
        compare_pair = masss.summarize_joint_samples

    else:
        summarize_one = masss.summarize_one_branch_samples_batch
        compare_pair = masss.summarize_joint_samples_batch

    return {
        'individual': {
            b: summarize_one(
                samples[b],
                quantiles=individual_summary_quantiles
            ) for b in branch_list
        },
        'comparative': {
            b: compare_pair(
                samples[b], samples[ref_branch_label],
                quantiles=comparative_summary_quantiles
            ) for b in set(branch_list) - {ref_branch_label}
        },
    }


def bootstrap_one_branch(
    sc, data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=masss.default_quantiles
):
    """Bootstrap on the means of one branch on its own.

    Generates `num_samples` sampled means, then returns summary
    statistics for their distribution.

    Args:
        sc: The spark context
        data: The data as a list, 1D numpy array, or pandas Series
        stat_fn: Either a function that aggregates each resampled
            population to a scalar (e.g. the default value `np.mean`
            lets you bootstrap means), or a function that aggregates
            each resampled population to a dict of scalars. In both
            cases, this function must accept a one-dimensional ndarray
            as its input.
        num_samples: The number of bootstrap iterations to perform
        seed_start: An int with which to seed numpy's RNG. It must
            be unique within this set of calculations.
        threshold_quantile: TODO
    """
    samples = get_bootstrap_samples(
        sc, data, stat_fn, num_samples, seed_start, threshold_quantile
    )

    # Depending on whether `stat_fn` returns a scalar or a dict,
    # we might have to call the 'batch' versions:
    if isinstance(samples, pd.Series):
        return masss.summarize_one_branch_samples(
            samples, quantiles=summary_quantiles
        )
    else:
        return masss.summarize_one_branch_samples_batch(
            samples, quantiles=summary_quantiles
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
    """Return samples of `stat_fn` evaluated on resampled data.

    Do the resampling in parallel over the cluster.

    FIXME: decide which of the following forms of stat_fn to support:
    1. Function that maps 1D array -> scalar
    2. Function that maps 1D array -> dict of scalars
    3. Dict of functions that each map 1D array -> scalar
    Almost certainly will support (1). But which others?
    (3) seems neater/easier to reason about than (2), but (2) will
    probably be more efficient when bootstrapping over many quantiles,
    since calling `np.quantile(..., [0.01, 0.02, ...])` once will be
    faster calling `np.quantile(..., 0.01)`, `np.quantile(..., 0.02)`
    etc. Could possibly support both (2) and (3) but ergh.
    A related question is whether the return values of (2) and (3)
    are dicts or ndarrays.

    Args:
        sc: The spark context
        data: The data as a list, 1D numpy array, or pandas series
        stat_fn: Either a function that aggregates each resampled
            population to a scalar (e.g. the default value `np.mean`
            lets you bootstrap means), or a function that aggregates
            each resampled population to a dict of scalars. In both
            cases, this function must accept a one-dimensional ndarray
            as its input.
        num_samples: The number of samples to return
        seed_start: A seed for the random number generator; this
            function will use seeds in the range
                [seed_start, seed_start + num_samples)
            and these particular seeds must not be used elsewhere
            in this calculation. By default, use a random seed.

    Returns:
        - By default, a pandas Series of sampled means
        - if `stat_fn` returns a scalar, a pandas Series
        - if `stat_fn` returns a dict, a pandas DataFrame with
            columns set to the dict keys.
    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    if threshold_quantile:
        data = _filter_outliers(data, threshold_quantile)

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
        return summary_df.iloc[:, 0]

    # Else return a DataFrame if stat_fn returns a dict
    return summary_df


def get_bootstrap_samples_local(data, stat_fn, num_samples):
    """Equivalent to `get_bootstrap_samples` but doesn't require Spark.

    The main purpose of this function is to document what's being done
    in `get_bootstrap_samples` :D FIXME: or maybe running when spark
    isn't available? Do we actually want to make this function work?
    """
    return [
        _resample_and_agg_once(stat_fn, data)  # slower?
        # stat_fn(np.random.choice(data, size=len(data)))
        for _ in range(num_samples)
    ]


# Functions that calculate stats over one resampled population


def _resample_and_agg_once_bcast(broadcast_data, stat_fn, unique_seed=None):
    np.random.seed(unique_seed)
    return _resample_and_agg_once(broadcast_data.value, stat_fn)


def _resample_and_agg_once(data, stat_fn):
    n = len(data)
    # TODO: can't we just use np.random.choice? Wouldn't that be faster?
    # There's not thaaat much difference in RAM requirements?
    randints = np.random.randint(0, n, n)
    resampled_data = data[randints]

    return stat_fn(resampled_data)


# Utility functions


def _filter_outliers(branch_data, threshold_quantile):
    """Return branch_data with outliers removed.

    N.B. `branch_data` is for an individual branch: if you do it for
    the entire experiment population in whole, then you may bias the
    results.

    FIXME: here we remove outliers - should we have an option or
    default to cap them instead?

    Args:
        branch_data: Data for one branch as a 1D ndarray or similar.
        threshold_quantile (float): Discard outliers above this
            quantile.

    Returns the subset of branch_data that was at or below the threshold
    quantile.
    """
    if threshold_quantile >= 1 or threshold_quantile < 0.5:
        raise ValueError("'threshold_quantile' should be close to 1")

    threshold_val = np.quantile(branch_data, threshold_quantile)

    return branch_data[branch_data <= threshold_val]
