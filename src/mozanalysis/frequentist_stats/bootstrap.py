# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats as mabs
from mozanalysis.utils import filter_outliers
from typing import cast
import mozanalysis.types as types


def compare_branches(
    df: pd.DataFrame,
    col_label: str,
    ref_branch_label: types.BranchLabel = "control",
    stat_fn: types.StatFunctionType = np.mean,
    num_samples: int = 10000,
    threshold_quantile: float | None = None,
    individual_summary_quantiles: types.QuantilesType = mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles: types.QuantilesType = mabs.DEFAULT_QUANTILES,
) -> types.CompareBranchesOutput | types.ParameterizedCompareBranchesOutput:
    """Jointly sample bootstrapped statistics then compare them.

    Performs a percentile bootstrap, which, according to Efron,
    is not significantly more distasteful than a basic bootstrap,
    regardless of what you may read on Stack Overflow.

    Args:
        df: a pandas DataFrame of queried experiment data in the
            standard format (see ``mozanalysis.experiment``).
        col_label (str or list): Label for the df column contaning the metric
            to be analyzed. If a list, labels for the multiple metrics to be analyzed.
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
    branch_list = cast(pd.Series[types.BranchLabel], df.branch.unique())

    if ref_branch_label not in branch_list:
        raise ValueError(
            f"Branch label '{ref_branch_label}' not in branch list '{branch_list}"
        )

    samples = cast(
        types.AnySamplesByBranch,
        {
            # TODO: do we need to control seed_start? If so then we must be careful here
            b: get_bootstrap_samples(
                df[col_label][df.branch == b],
                stat_fn,
                num_samples,
                threshold_quantile=threshold_quantile,
            )
            for b in branch_list
        },
    )

    return mabs.compare_samples(
        samples,
        ref_branch_label,
        individual_summary_quantiles,
        comparative_summary_quantiles,
    )


def bootstrap_one_branch(
    data: pd.Series[types.Numeric],
    stat_fn: types.StatFunctionType = np.mean,
    num_samples: int = 10000,
    seed_start: int | None = None,
    threshold_quantile: float | None = None,
    summary_quantiles: types.QuantilesType = mabs.DEFAULT_QUANTILES,
) -> types.Estimates | types.ParameterizedEstimates:
    """Run a bootstrap for one branch on its own.

    Resamples the data ``num_samples`` times, computes ``stat_fn`` for
    each sample, then returns summary statistics for the distribution
    of the outputs of ``stat_fn``.

    Args:
        data: The data as a 1D numpy array, pandas series, or pandas dataframe.
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
    samples = get_bootstrap_samples(
        data, stat_fn, num_samples, seed_start, threshold_quantile
    )

    return mabs.summarize_one_branch_samples(samples, summary_quantiles)


def get_bootstrap_samples(
    data: pd.Series[types.Numeric] | pd.DataFrame | types.NumericNDArray,
    stat_fn: types.StatFunctionType = np.mean,
    num_samples: int = 10000,
    seed_start: int | None = None,
    threshold_quantile: float | None = None,
) -> types.BootstrapSamples | types.ParameterizedBootstrapSamples:
    """Return ``stat_fn`` evaluated on resampled and original data.

    Do the resampling in parallel over the cluster.

    Args:
        data: The data as a 1D numpy array, pandas series, or pandas dataframe.
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
        ``stat_fn`` evaluated over ``num_samples`` samples.

            * By default, a pandas Series of sampled means
            * if ``stat_fn`` returns a scalar, a pandas Series
            * if ``stat_fn`` returns a dict, a pandas DataFrame
              with columns set to the dict keys.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = np.array(data.to_numpy(dtype="float", na_value=np.nan))

    data = cast(types.NumericNDArray, data)

    if np.isnan(data).any():
        raise ValueError("'data' contains null values")

    if threshold_quantile:
        data = filter_outliers(data, threshold_quantile)

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every call has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_samples)

    summary_stat_samples = [
        _resample_and_agg_once(data, stat_fn, unique_seed) for unique_seed in seed_range
    ]

    summary_df = pd.DataFrame(summary_stat_samples)
    if len(summary_df.columns) == 1:
        # Return a Series if stat_fn returns a scalar
        return summary_df.iloc[:, 0]

    # Else return a DataFrame if stat_fn returns a dict
    return summary_df


def _resample_and_agg_once(
    data: types.NumericNDArray,
    stat_fn: types.StatFunctionType,
    unique_seed: int | None = None,
) -> types.StatFunctionReturnType:
    random_state = np.random.RandomState(unique_seed)

    n = len(data)
    # TODO: can't we just use random_state.choice? Wouldn't that be faster?
    # There's not thaaat much difference in RAM requirements?
    randints = random_state.randint(0, n, n)
    resampled_data = data[randints]

    return stat_fn(resampled_data)


def compare_branches_quantiles(
    df: pd.DataFrame,
    col_label: str,
    ref_branch_label: types.BranchLabel = "control",
    quantiles_of_interest: list[float] | None = None,
    num_samples: int = 10000,
    threshold_quantile: float | None = None,
    individual_summary_quantiles: types.QuantilesType = mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles: types.QuantilesType = mabs.DEFAULT_QUANTILES,
) -> types.ParameterizedCompareBranchesOutput:
    """
    Performs inferences on the metric quantiles inspired by Spotify's
    "Resampling-free bootstrap inference for quantiles" approach
    https://arxiv.org/pdf/2202.10992.pdf.

    Parameters are similar to `compare_branches` except for:

    Args:
        quantiles (List[float]): a list of quantiles upon which inferences are desired.
        Ex: 0.2 is the 20th percentile, 0.5 is the median, etc.
    """

    if quantiles_of_interest is None:
        quantiles_of_interest = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    branch_list = cast(pd.Series[types.BranchLabel], df.branch.unique())

    if ref_branch_label not in branch_list:
        raise ValueError(
            f"Branch label '{ref_branch_label}' not in branch list '{branch_list}"
        )

    samples = {
        b: get_quantile_bootstrap_samples(
            df[col_label][df.branch == b],
            quantiles_of_interest,
            num_samples,
            threshold_quantile=threshold_quantile,
        )
        for b in branch_list
    }

    return cast(
        types.ParameterizedCompareBranchesOutput,
        mabs.compare_samples(
            samples,
            ref_branch_label,
            individual_summary_quantiles,
            comparative_summary_quantiles,
        ),
    )


def get_quantile_bootstrap_samples(
    data: "pd.Series[types.Numeric]",
    quantiles_of_interest: list[float],
    num_samples: int = 10000,
    threshold_quantile: float | None = None,
) -> types.ParameterizedBootstrapSamples:
    """Params are similar to `get_bootstrap_samples`"""
    # if type(data) is not np.ndarray:
    data_arr = cast(
        types.NumericNDArray, np.array(data.to_numpy(dtype="float", na_value=np.nan))
    )

    if np.isnan(data_arr).any():
        raise ValueError("'data' contains null values")

    if threshold_quantile:
        data_arr = filter_outliers(data_arr, threshold_quantile)

    data_arr = cast(types.NumericNDArray, np.sort(data_arr))

    sample_size = data_arr.shape[0]
    samples = {
        f"{quantile:.1}": data_arr[
            np.random.binomial(sample_size - 1, quantile, num_samples)
        ]
        for quantile in quantiles_of_interest
    }
    df = pd.DataFrame.from_dict(samples)

    return df
