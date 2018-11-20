# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import mock
import numpy as np
from pytest import approx

from mozanalysis.stats import bootstrap, _t_percentile


# Generate a normalized distribution around a known mean.
MEAN = 100.0
STDDEV = 10.0
TOLERANCE = 0.1
DATA = np.random.normal(MEAN, STDDEV, 1000)


def test_t_percentile():
    t = np.mean(DATA)
    d = _t_percentile(DATA, 0.95, t)
    low = np.mean(DATA) - 1.96 * STDDEV
    high = np.mean(DATA) + 1.96 * STDDEV
    assert approx(low, rel=TOLERANCE) == d["confidence_low"]
    assert approx(high, rel=TOLERANCE) == d["confidence_high"]


def test_bootstrap_mean(spark_context):
    d = bootstrap(
        spark_context, DATA, np.mean, num_iterations=50, confidence_level=0.95
    )
    # Check that the calculated mean from the samples are in expected range.
    assert approx(MEAN, rel=TOLERANCE) == d["calculated_value"]

    # Check that our confidence values are within expected ranges.
    assert approx(MEAN * 0.975, rel=TOLERANCE) == d["confidence_low"]
    assert approx(MEAN * 1.025, rel=TOLERANCE) == d["confidence_high"]


@mock.patch("mozanalysis.stats._t_percentile")
def test_percentile_data(mocked, spark_context):
    # Test that we pass the bootstrap data to `_t_percentile`.
    data = np.arange(20)
    bootstrap(spark_context, data, np.mean, num_iterations=5)
    assert list(mocked.call_args[0][0]) != list(data)
