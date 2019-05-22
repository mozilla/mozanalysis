# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd


default_quantiles = (0.005, 0.05, 0.5, 0.95, 0.995)


def compare_two_sample_sets(focus, reference, quantiles=default_quantiles):
    # Primary use case is comparing two sample sets
    if len(focus.shape) == 1:
        return _compare_two_sample_sets(focus, reference, quantiles)

    # It's common that we'll want to do this for a batch of sample sets,
    # each representing a different bootstrap stat_fn, or time period,
    # or threshold.
    if set(focus.columns) != set(reference.columns):
        raise ValueError()

    return pd.DataFrame({
        k: _compare_two_sample_sets(focus[k], reference[k], quantiles)
        for k in focus.columns
    }, columns=focus.columns)


def _compare_two_sample_sets(focus, reference, quantiles):
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


def summarize_one_sample_set(data, quantiles=default_quantiles):
    return data.agg(_summarize_one_sample_set, quantiles=quantiles)


def _summarize_one_sample_set(data, quantiles):
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        # Hey pd.Series.agg - don't apply me elementwise!
        raise TypeError("Can't summarize a scalar")

    q_index = [str(v) for v in quantiles]

    res = pd.Series(index=q_index + ['mean'])

    res[q_index] = np.quantile(data, quantiles)
    res['mean'] = np.mean(data)
    return res
