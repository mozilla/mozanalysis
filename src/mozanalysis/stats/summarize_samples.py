# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd


# TODO: is this kind of standardization helpful?
# Will it discourage Bonferroni corrections? Is that a major problem?

res_columns = [
    'rel_uplift_0.005',
    'rel_uplift_0.05',
    'rel_uplift_0.5',
    'rel_uplift_0.95',
    'rel_uplift_0.995',
    'rel_uplift_exp',
    'abs_uplift_0.005',
    'abs_uplift_0.05',
    'abs_uplift_0.5',
    'abs_uplift_0.95',
    'abs_uplift_0.995',
    'abs_uplift_exp',
    'max_abs_diff_0.95',
]


one_res_index = [
    '0.005',
    '0.05',
    'mean',
    '0.95',
    '0.995',
]

# TODO: use dictionaries instead of Series?


def compare_two_sample_sets(focus, reference):
    res = pd.Series(index=res_columns)

    rel_uplift_samples = focus / reference - 1

    res[
        'rel_uplift_0.005':'rel_uplift_0.995'
    ] = np.quantile(rel_uplift_samples, [0.005, 0.05, 0.5, 0.95, 0.995])
    res['rel_uplift_exp'] = np.mean(rel_uplift_samples)

    abs_uplift_samples = focus - reference

    res[
        'abs_uplift_0.005':'abs_uplift_0.995'
    ] = np.quantile(abs_uplift_samples, [0.005, 0.05, 0.5, 0.95, 0.995])
    res['abs_uplift_exp'] = np.mean(abs_uplift_samples)

    res['max_abs_diff_0.95'] = np.quantile(np.abs(abs_uplift_samples), 0.95)

    res['prob_win'] = np.mean(focus > reference)

    return res


def summarize_one_sample_set(data):
    res = pd.Series(index=one_res_index)
    res['mean'] = np.mean(data)
    quantiles = [0.005, 0.05, 0.95, 0.995]
    res[[str(v) for v in quantiles]] = np.quantile(data, quantiles)
    return res
