# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats as mabs
from mozanalysis.utils import filter_outliers


def compare_branches(
    df, col_label, ref_branch_label='control', stat_fn=np.mean,
    num_samples=10000, threshold_quantile=None,
    individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles=mabs.DEFAULT_QUANTILES, sc=None
):
    """Jointly sample bootstrapped statistics then compare them.

    Performs a percentile bootstrap, which, according to Efron,
    is not significantly more distasteful than a basic bootstrap,
    regardless of what you may read on Stack Overflow.

    Args:
        df: a pandas DataFrame of queried experiment data in the
            standard format (see ``mozanalysis.experiment``).
        col_label (str): Label for the df column contaning the metric
            to be analyzed.
        ref_branch_label (str, optional): String in ``df['branch']`` that
            identifies the branch with respect to which we want to
            calculate uplifts - usually the control branch.
        stat_fn (func, optional): A function that either:

            - Aggregates each resampled population to a scalar (e.g.
                the default, ``np.mean``), or
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
        sc (optional): The Spark context, if available

    Returns a dictionary:
        If ``stat_fn`` returns a scalar (this is the default), then
        this function returns a dictionary has the following keys and
        values:

            'individual': dictionary mapping each branch name to a pandas
                Series that holds the expected value for the bootstrapped
                ``stat_fn``, and confidence intervals.
            'comparative': dictionary mapping each branch name to a pandas
                Series of summary statistics for the possible uplifts of
                the bootstrapped ``stat_fn`` relative to the reference branch.

        Otherwise, when ``stat_fn`` returns a dict, then this function
        returns a similar dictionary, except the Series are replaced with
        DataFrames. Each row in each DataFrame corresponds to one output
        of `stat_fn`, and is the Series that would be returned if ``stat_fn``
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
            df[col_label][df.branch == b],
            stat_fn,
            num_samples,
            threshold_quantile=threshold_quantile,
            sc=sc,
        ) for b in branch_list
    }

    return mabs.compare_samples(
        samples, ref_branch_label, individual_summary_quantiles,
        comparative_summary_quantiles
    )


def bootstrap_one_branch(
    data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=mabs.DEFAULT_QUANTILES,
    sc=None
):
    """Run a bootstrap for one branch on its own.

    Resamples the data ``num_samples`` times, computes ``stat_fn`` for
    each sample, then returns summary statistics for the distribution
    of the outputs of ``stat_fn``.

    Args:
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
        sc (optional): The Spark context, if available
    """
    samples = get_bootstrap_samples(
        data, stat_fn, num_samples, seed_start, threshold_quantile, sc
    )

    return mabs.summarize_one_branch_samples(
        samples, summary_quantiles
    )


def get_bootstrap_samples(
    data, stat_fn=np.mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, sc=None
):
    """Return ``stat_fn`` evaluated on resampled and original data.

    Do the resampling in parallel over the cluster.

    Args:
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
        sc (optional): The Spark context, if available

    Returns:
        ``stat_fn`` evaluated over ``num_samples`` samples.

            * By default, a pandas Series of sampled means
            * if ``stat_fn`` returns a scalar, a pandas Series
            * if ``stat_fn`` returns a dict, a pandas DataFrame
              with columns set to the dict keys.
    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    if np.isnan(data).any():
        raise ValueError("'data' contains null values")

    if threshold_quantile:
        data = filter_outliers(data, threshold_quantile)

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every call has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_samples)

    if sc is None:
        summary_stat_samples = [
            _resample_and_agg_once(data, stat_fn, unique_seed)
            for unique_seed in seed_range
        ]

    else:
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
