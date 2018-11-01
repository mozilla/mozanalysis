# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import mock
import numpy as np

from mozanalysis.stats import bootstrap, _percentile


# Generate a normalized distribution around a known mean.
MEAN = 15.0
TOLERANCE = 0.2
STDDEV = 0.1
DATA = np.random.normal(MEAN, STDDEV, 1000)


def test_percentile():
    data = np.arange(10)
    p = _percentile(data, 0.8)
    assert round(p["confidence_low"], 1) == 0.9
    assert round(p["confidence_high"], 1) == 8.1


def test_mean():
    assert MEAN - TOLERANCE < np.mean(DATA) < MEAN + TOLERANCE


def test_bootstrap_mean(spark_context):
    d = bootstrap(
        spark_context, DATA, np.mean, num_iterations=50, confidence_level=0.95
    )
    # Check that the calculated mean from the samples are in expected range.
    assert MEAN - TOLERANCE < d["calculated_value"] < MEAN + TOLERANCE
    # Check that our confidence values are within expected ranges.
    low = np.mean(DATA) - 1.96 * STDDEV
    high = np.mean(DATA) + 1.96 * STDDEV
    assert low - TOLERANCE < d["confidence_low"] < low + TOLERANCE
    assert high - TOLERANCE < d["confidence_high"] < high + TOLERANCE


@mock.patch("mozanalysis.stats._percentile")
def test_percentile_data(mocked, spark_context):
    # Test that we pass the bootstrap data to `_percentile`.
    data = np.arange(20)
    bootstrap(spark_context, data, np.mean, num_iterations=5)
    assert list(mocked.call_args[0][0]) != list(data)
