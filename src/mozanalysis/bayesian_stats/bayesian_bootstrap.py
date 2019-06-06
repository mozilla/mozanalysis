import numpy as np
import pandas as pd

import mozanalysis.bayesian_stats as mabs
from mozanalysis.utils import filter_outliers


def bb_mean(data, prob_weights):
    return np.dot(data, prob_weights)


def make_bb_quantile_closure(quantiles):
    def bb_quantile(data, prob_weights):
        # assume data is previously sorted, as per np.unique()
        cdf = np.cumsum(prob_weights)

        res = np.interp(quantiles, data, cdf)

        if np.isscalar(quantiles):
            return res

        else:
            return dict(zip(quantiles, res))

    return bb_quantile


def compare_branches(
    sc, df, col_label, ref_branch_label='control', stat_fn=bb_mean,
    num_samples=10000, threshold_quantile=None,
    individual_summary_quantiles=mabs.DEFAULT_QUANTILES,
    comparative_summary_quantiles=mabs.DEFAULT_QUANTILES
):
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

    return {
        'individual': {
            b: mabs.summarize_one_branch_samples(
                samples[b],
                quantiles=individual_summary_quantiles
            ) for b in branch_list
        },
        'comparative': {
            b: mabs.summarize_joint_samples(
                samples[b], samples[ref_branch_label],
                quantiles=comparative_summary_quantiles
            ) for b in set(branch_list) - {ref_branch_label}
        },
    }


def bootstrap_one_branch(
    sc, data, stat_fn=bb_mean, num_samples=10000, seed_start=None,
    threshold_quantile=None, summary_quantiles=mabs.DEFAULT_QUANTILES
):
    samples = get_bootstrap_samples(
        sc, data, stat_fn, num_samples, seed_start, threshold_quantile
    )

    if isinstance(samples, pd.Series):
        return mabs.summarize_one_branch_samples(
            samples, summary_quantiles
        )
    else:
        return mabs.summarize_one_branch_samples_batch(
            samples, summary_quantiles
        )


def get_bootstrap_samples(
    sc, data, stat_fn=bb_mean, num_samples=10000, seed_start=None,
    threshold_quantile=None
):
    if not type(data) == np.ndarray:
        data = np.array(data)

    if threshold_quantile:
        data = filter_outliers(data, threshold_quantile)

    data_values, data_counts = np.unique(data, return_counts=True)

    if seed_start is None:
        seed_start = np.random.randint(np.iinfo(np.uint32).max)

    # Deterministic "randomness" requires careful state handling :(
    # Need to ensure every call has a unique, deterministic seed.
    seed_range = range(seed_start, seed_start + num_samples)

    # TODO: run locally `if sc is None`?
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
