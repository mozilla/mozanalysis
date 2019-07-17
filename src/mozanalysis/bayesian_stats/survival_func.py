# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats.binary as mabsb


def compare_branches(df, col_label, ref_branch_label='control', thresholds=None):
    """Return the survival functions and relative uplifts thereupon.

    This function generates data for a metric's survival function
    (1 - cumulative distribution function) for each branch, and
    calculates the relative uplift compared to the reference branch
    identified by ``ref_branch_label``.

    It converts the non-negative, real-valued per-user metric data in
    ``df[col_label]`` into ``n=len(thresholds)`` different binary
    metrics, and analyzes these ``n`` metrics with the Bayesian binary
    methods.

    The precise values of the thresholds usually don't matter unless
    certain thresholds have been standardized outside the context of
    this experiment.

    The results are related to those obtained by bootstrapping a range
    of quantiles over the data:

    * In the survival plot, we set a value for the metric and calculate
      the fraction of the data that was above this value, with
      uncertainty on the fraction.
    * When bootstrapping quantiles, we set a quantile (a fraction of
      the data) and find the value such that the given fraction of
      data is greater than this value, with uncertainty on the value.

    Reiterating: if we plot the survival function with metric values
    on the x axis and "fractions" on the y axis, then this function
    first chooses some sensible values for x then runs statistics to
    compute values for y, with uncertainty. If we were bootstrapping
    quantiles, then we would choose some sensible values for y then
    run statistics to compute values for x, with uncertainty.

    Args:
        df: a pandas DataFrame of queried experiment data in the
            standard format. Target metric should be non-negative.
        col_label (str): Label for the df column contaning the metric
            to be analyzed.
        ref_branch_label (str, optional): String in ``df['branch']``
            that identifies the the branch with respect to which we
            want to calculate uplifts - usually the control branch.
        thresholds (list/ndarray, optional): Thresholds that define the
            metric's quantization; ``df[col_label]``

    Returns a dictionary:

        * 'individual': dictionary mapping branch names to a pandas
          DataFrame containing values from the survival function.
          The DataFrames' indexes are the list of thresholds; the
          columns are summary statistics on the survival function.
        * 'comparative': dictionary mapping branch names to a pandas
          DataFrame of summary statistics for the possible uplifts of the
          conversion rate relative to the reference branch - see docs
          for
          :meth:`mozanalysis.stats.summarize_samples.summarize_joint_samples_batch`.
    """
    branch_list = df.branch.unique()

    if not thresholds:
        thresholds = get_thresholds(df[col_label])

    data = {t: _one_thresh(t, df, col_label, ref_branch_label) for t in thresholds}

    return {
        'individual': {
            b: pd.DataFrame({
                t: d['individual'][b] for t, d in data.items()
            }, columns=thresholds).T
            for b in branch_list
        },
        'comparative': {
            b: pd.DataFrame({
                t: d['comparative'][b] for t, d in data.items()
            }).T for b in set(branch_list) - {ref_branch_label}
        },
    }


def get_thresholds(col, max_num_thresholds=101):
    """Return a set of interesting thresholds for the dataset ``col``

    Assumes that the values are non-negative, with zero as a special case.

    Args:
        col: a Series of individuals' data for a metric
        max_num_thresholds (int): Return at most this many threshold values.

    Returns:
        A list of thresholds. By default these are de-duped percentiles
        of the nonzero data.
    """
    if col.isnull().any():
        raise ValueError("'col' contains null values")

    if col.min() < 0:
        raise ValueError("This function assumes non-negative data")

    # When taking quantiles, treat "0" as a special case so that we
    # still have resolution if 99% of users are 0.
    nonzero_quantiles = col[col > 0].quantile(
        np.linspace(0, 1, max_num_thresholds),
        # 'nearest' is not what we want, but is the least-bad option.
        # Can't use the default 'linear' because we want to call 'unique()'
        # to avoid duplicating work.
        # Can't use 'lower', 'higher', or 'midpoint' due to rounding issues
        # that lead to dumb choices. That leaves us with 'nearest'
        interpolation='nearest'
    )
    return sorted(
        [np.float64(0)] + list(nonzero_quantiles.unique())
    )[:-1]  # The thresholds get used as `>` not `>=`, so exclude the max value


def _one_thresh(threshold, df, col_label, ref_branch_label):
    """Run stats on the fraction of clients above ``threshold``."""
    if df[col_label].isnull().any():
        raise ValueError("'df' contains null values for '{}'".format(col_label))

    if '_tmp_threshold_val' in df.columns:
        raise ValueError(
            "Either you have an exceedingly poor taste in column names, "
            "or there is a bug in `_one_thresh`."
        )
    try:
        # Sorry for mutating the input inplace. I'll be sure to tidy up.
        df['_tmp_threshold_val'] = df[col_label] > threshold

        return mabsb.compare_branches(
            df, '_tmp_threshold_val', ref_branch_label=ref_branch_label
        )

    finally:
        df.drop('_tmp_threshold_val', axis='columns', inplace=True)
