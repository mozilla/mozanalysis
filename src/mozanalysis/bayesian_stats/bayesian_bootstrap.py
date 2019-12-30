# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats as mabs
from mozanalysis.utils import filter_outliers


def bb_mean(values, prob_weights):
    """Calculate the mean of a bootstrap replicate.

    Args:
        values (pd.Series, ndarray): One dimensional array
            of observed values
        prob_weights (pd.Series, ndarray): Equally shaped
            array of the probability weight associated with
            each value.

    Returns:
        The mean as a np.float.
    """
    return np.dot(values, prob_weights)


def make_bb_quantile_closure(quantiles):
    """Return a function to calculate quantiles for a bootstrap replicate.

    Args:
        quantiles (float, list of floats): Quantiles to compute

    Returns a function that calculates quantiles for a bootstrap replicate:

        Args:

            values (pd.Series, ndarray):
                One dimensional array of observed values
            prob_weights (pd.Series, ndarray):
                Equally shaped array of the probability weight associated with
                each value.

        Returns:

            * A quantile as a np.float, or
            * several quantiles as a dict keyed by the quantiles

    """

    # If https://github.com/numpy/numpy/pull/9211/ is ever merged then
    # we can just use that instead.

    def get_value_at_quantile(values, cdf, quantile):
        """Return the value at a quantile.

        Does no interpolation because our Bayesian bootstrap
        implementation calls `np.unique` to tally the values:
        if it did not take this shortcut then regardless of whether
        we interpolate when returning quantiles, the vast majority
        of quantiles would coincide with a value. But since we take
        this shortcut, interpolation mostly returns values not in the
        dataset. Ergh.
        """
        # Add a tolerance of 1e-6 to account for numerical error when
        # computing the cdf
        arg = np.nonzero(quantile < cdf + 1e-6)[0][0]
        return values[arg]

    def bb_quantile(values, prob_weights):
        """Calculate quantiles for a bootstrap replicate.

        Args:
            values (pd.Series, ndarray): One dimensional array
                of observed values
            prob_weights (pd.Series, ndarray): Equally shaped
                array of the probability weight associated with
                each value.

        Returns:

            * A quantile as a np.float, or
            * several quantiles as a dict keyed by the quantiles
        """
        # assume values is previously sorted, as per np.unique()
        cdf = np.cumsum(prob_weights)

        if np.isscalar(quantiles):
            return get_value_at_quantile(values, cdf, quantiles)

        else:
            return {
                q: get_value_at_quantile(values, cdf, q)
                for q in quantiles
            }

    return bb_quantile


def compare_branches(
    df, col_label, ref_branch_label='control', stat_fn=bb_mean,
    num_samples=10000, threshold_quantile=None,
    individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles=mabs.DEFAULT_QUANTILES,
    sc=None
):
    """Jointly sample bootstrapped statistics then compare them.

    Args:
        df: a pandas DataFrame of queried experiment data in the
            standard format (see `mozanalysis.experiment`).
        col_label (str): Label for the df column contaning the metric
            to be analyzed.
        ref_branch_label (str, optional): String in ``df['branch']``
            that identifies the branch with respect to which we want to
            calculate uplifts - usually the control branch.
        stat_fn (callable, optional): A function that either:

            * Aggregates each resampled population to a scalar (e.g.
              the default, ``bb_mean``), or
            * Aggregates each resampled population to a dict of
              scalars (e.g. the func returned by
              ``make_bb_quantile_closure`` when given multiple
              quantiles.

            In both cases, this function must accept two parameters:

            * a one-dimensional ndarray or pandas Series of values,
            * an identically shaped object of weights for these values

        num_samples (int, optional): The number of bootstrap iterations
            to perform.
        threshold_quantile (float, optional): An optional threshold
            quantile, above which to discard outliers. E.g. `0.9999`.
        individual_summary_quantiles (list, optional): Quantiles to
            determine the credible intervals on individual branch
            statistics. Change these when making Bonferroni corrections.
        comparative_summary_quantiles (list, optional): Quantiles to
            determine the credible intervals on comparative branch
            statistics (i.e. the change relative to the reference
            branch, probably the control). Change these when making
            Bonferroni corrections.
        sc (optional): The Spark context, if available

    Returns:
        If ``stat_fn`` returns a scalar (this is the default), then
        this function returns a dictionary has the following keys and
        values:

            * 'individual': dictionary mapping each branch name to a pandas
              Series that holds the expected value for the bootstrapped
              ``stat_fn``, and credible intervals.
            * 'comparative': dictionary mapping each branch name to a pandas
              Series of summary statistics for the possible uplifts of
              the bootstrapped ``stat_fn`` relative to the reference branch.

        Otherwise, when ``stat_fn`` returns a dict, then this function
        returns a similar dictionary, except the Series are replaced with
        DataFrames. Each row in each DataFrame corresponds to one output
        of ``stat_fn``, and is the Series that would be returned if
        ``stat_fn`` computed only this statistic.
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
    data, stat_fn=bb_mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=mabs.DEFAULT_QUANTILES,
    sc=None
):
    """Bootstrap ``stat_fn`` for one branch on its own.

    Computes ``stat_fn`` for ``num_samples`` resamples of ``data``,
    then returns summary statistics for the results.

    Args:
        data: The data as a list, 1D numpy array, or pandas Series
        stat_fn (callable, optional): A function that either:

            * Aggregates each resampled population to a scalar (e.g.
              the default, ``bb_mean``), or
            * Aggregates each resampled population to a dict of
              scalars (e.g. the func returned by
              ``make_bb_quantile_closure`` when given multiple
              quantiles.

            In both cases, this function must accept two parameters:

            * a one-dimensional ndarray or pandas Series of values,
            * an identically shaped object of weights for these values

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

    return mabs.summarize_one_branch_samples(samples, summary_quantiles)


def get_bootstrap_samples(
    data, stat_fn=bb_mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, sc=None
):
    """Return ``stat_fn`` evaluated on resampled data.

    Args:
        data: The data as a list, 1D numpy array, or pandas series
        stat_fn (callable, optional): A function that either:

            * Aggregates each resampled population to a scalar (e.g.
              the default, ``bb_mean``), or
            * Aggregates each resampled population to a dict of
              scalars (e.g. the func returned by
              ``make_bb_quantile_closure`` when given multiple
              quantiles.

            In both cases, this function must accept two parameters:

            * a one-dimensional ndarray or pandas Series of values,
            * an identically shaped object of weights for these values

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
        A Series or DataFrame with one row per sample and one column
        per output of ``stat_fn``.

    References:
        Rubin, Donald B. The Bayesian Bootstrap. Ann. Statist. 9 (1981),
            no. 1, 130--134. https://dx.doi.org/10.1214/aos/1176345338
    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    if np.isnan(data).any():
        raise ValueError("'data' contains null values")

    if threshold_quantile:
        data = filter_outliers(data, threshold_quantile)

    # For computational efficiency, tally the data. If you are careful
    # with the resulting draws from the dirichlet then this should be
    # equivalent to not doing this step (and passing np.ones() as the
    # counts)
    data_values, data_counts = np.unique(data, return_counts=True)

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every call has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_samples)

    if sc is None:
        summary_stat_samples = [
            _resample_and_agg_once(data_values, data_counts, stat_fn, unique_seed)
            for unique_seed in seed_range
        ]

    else:
        try:
            broadcast_data_values = sc.broadcast(data_values)
            broadcast_data_counts = sc.broadcast(data_counts)

            summary_stat_samples = sc.parallelize(seed_range).map(
                lambda seed: _resample_and_agg_once_bcast(
                    broadcast_data_values=broadcast_data_values,
                    broadcast_data_counts=broadcast_data_counts,
                    stat_fn=stat_fn,
                    unique_seed=seed % np.iinfo(np.uint32).max,
                )
            ).collect()

        finally:
            broadcast_data_values.unpersist()
            broadcast_data_counts.unpersist()

    summary_df = pd.DataFrame(summary_stat_samples)
    if len(summary_df.columns) == 1:
        # Return a Series if stat_fn returns a scalar
        return summary_df.iloc[:, 0]

    # Else return a DataFrame if stat_fn returns a dict
    return summary_df


def _resample_and_agg_once_bcast(
    broadcast_data_values, broadcast_data_counts, stat_fn, unique_seed
):
    return _resample_and_agg_once(
        broadcast_data_values.value, broadcast_data_counts.value,
        stat_fn, unique_seed
    )


def _resample_and_agg_once(
    data_values, data_counts, stat_fn, unique_seed=None
):
    random_state = np.random.RandomState(unique_seed)

    prob_weights = random_state.dirichlet(data_counts)

    return stat_fn(data_values, prob_weights)
