# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
from functools import partial


def _resample(iteration, stat_fn, broadcast_data):
    np.random.seed(iteration)
    n = len(broadcast_data.value)
    randints = np.random.randint(0, n, n)
    return stat_fn(broadcast_data.value[randints])


def _percentile(data, confidence_level):
    """Returns low and high percentile based on provided confidence level"""
    confidence_margin = 0.5 * (1.0 - confidence_level)
    confidence_low = 0.0 + confidence_margin
    confidence_high = 1.0 - confidence_margin

    return {
        "confidence_low": np.percentile(data, confidence_low * 100),
        "confidence_high": np.percentile(data, confidence_high * 100),
    }


def bootstrap(sc, data, stat_fn, num_iterations=2000, confidence_level=0.95):
    """Returns a bootstrapped confidence interval

    Given an array of data, returns the bootstrapped confidence interval based
    on the confidence level provided. This uses pyspark and distributes the data
    to the nodes for efficiency.

    NOTE: All the data, plus one replicate, should fit into memory.

    Arguments:
    sc : The spark context.
    data : The data as a list or numpy array.
    stat_fn : A callable that computes the statistic.
    num_iterations : The number of bootstrap iterations to perform.
    confidence_level : The confidence level desired (between 0 and 1)

    Returns a dictionary with the keys::

        ["calcualted_value", "confidence_low", "confidence_high"]

    """
    if not type(data) == np.ndarray:
        data = np.array(data)

    # Broadcast the data as read-only to the clusters.
    broadcast_data = sc.broadcast(data)

    f = partial(_resample, stat_fn=stat_fn, broadcast_data=broadcast_data)
    stats = sc.parallelize(range(num_iterations)).map(f).collect()

    broadcast_data.unpersist()

    p = _percentile(stats, confidence_level)

    return {
        "calculated_value": stat_fn(data),
        "confidence_low": p["confidence_low"],
        "confidence_high": p["confidence_high"],
    }
