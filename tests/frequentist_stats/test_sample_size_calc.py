# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mozanalysis.frequentist_stats.sample_size import (
    z_or_t_ind_sample_size_calc,
    empirical_effect_size_sample_size_calc,
    sample_size_curves,
)
from mozanalysis.metrics.desktop import search_clients_daily, uri_count


@pytest.fixture
def fake_ts_result():
    class FakeTimeSeriesResult:
        def get_aggregated_data(self, metric_list, aggregate_function, **kwargs):
            periods = [0, 7, 14]
            if aggregate_function == "AVG":
                values = [5, 2, 1]
            else:
                values = [1, 2, 1]

            df = pd.DataFrame({"analysis_window_start": periods})
            for m in metric_list:
                df[m.name] = values
            return df, 1000

    return FakeTimeSeriesResult()


def test_sample_size_calc_desktop():
    df = pd.DataFrame(
        {
            search_clients_daily.name: np.random.normal(size=100),
            uri_count.name: np.random.normal(size=100),
        }
    )

    res = z_or_t_ind_sample_size_calc(df, [search_clients_daily, uri_count])

    assert all([c in res.keys() for c in df.columns])

    assert res[search_clients_daily.name]["sample_size_per_branch"] > 1000000
    assert res[uri_count.name]["sample_size_per_branch"] > 1000000


def test_empirical_effect_size_sample_size_calc(fake_ts_result):
    @dataclass
    class FakeMetric:
        name: str

    metric_list = [FakeMetric(name="metric1"), FakeMetric(name="metric2")]

    result = empirical_effect_size_sample_size_calc(
        res=fake_ts_result, bq_context=None, metric_list=metric_list
    )

    assert len(result) == 2
    for m in ["metric1", "metric2"]:
        r = result[m]
        assert r["effect_size"]["value"] == 3
        assert r["effect_size"]["period_start_day"] == 7
        assert r["mean"]["value"] == 5
        assert r["mean"]["period_start_day"] == 0
        assert r["std_dev"]["value"] == 2
        assert r["std_dev"]["period_start_day"] == 7
        assert r["relative_effect_size"] == 0.6
        assert r["number_of_clients_targeted"] == 1000
        np.testing.assert_allclose(r["sample_size_per_branch"], 8.44, atol=0.01)
        np.testing.assert_allclose(
            r["population_percent_per_branch"], 0.844, atol=0.001
        )


def test_emperical_effect_size_results_holder(fake_ts_result):
    """checks the get_dataframe method doesn't throw an error and has
    the expected columns in the case where a weekly mean is not generated"""

    @dataclass
    class FakeMetric:
        name: str

    metric_list = [FakeMetric(name="metric1"), FakeMetric(name="metric2")]

    result = empirical_effect_size_sample_size_calc(
        res=fake_ts_result, bq_context=None, metric_list=metric_list
    )

    result_df = result.get_dataframe()
    expected_columns = {
        "effect_size_value",
        "std_dev_value",
        "sample_size_per_branch",
        "population_percent_per_branch",
    }
    assert set(result_df.columns) == expected_columns


def test_curve_results_holder():
    """this test ensures the results_holder object has properly formatted attributes
    and is backward compatible"""
    df = pd.DataFrame(
        {
            search_clients_daily.name: np.random.normal(size=100),
            uri_count.name: np.random.normal(size=100),
        }
    )

    metrics = [search_clients_daily, uri_count]
    effect_size = np.arange(0.01, 0.11, 0.01)
    res = sample_size_curves(
        df,
        metrics,
        solver=z_or_t_ind_sample_size_calc,
        effect_size=effect_size,
        power=0.8,
        alpha=0.05,
        outlier_percentile=99.5,
    )
    metric_names = set([el.name for el in metrics])

    # set should only have 2 values: "search_clients_daily" "uri_count"
    assert len(res) == len(metric_names)
    assert set(res.keys()) == metric_names
    assert len(res.values()) == len(metric_names)
    iter_keys = []
    for key, val in res.items():
        iter_keys.append(key)
        assert set(val.columns) == {
            "effect_size",
            "sample_size_per_branch",
            "population_percent_per_branch",
            "number_of_clients_targeted",
        }
        assert set(val["effect_size"]) == set(effect_size)
    assert len(iter_keys) == len(metric_names)
    assert set(iter_keys) == metric_names


def test_curve_results_holder_pretty_df():
    """this test ensures the pretty_results function works as expected"""
    df = pd.DataFrame(
        {
            search_clients_daily.name: np.random.normal(size=100),
            uri_count.name: np.random.normal(size=100),
        }
    )

    metrics = [search_clients_daily, uri_count]
    effect_size = np.arange(0.01, 0.11, 0.01)
    res = sample_size_curves(
        df,
        metrics,
        solver=z_or_t_ind_sample_size_calc,
        effect_size=effect_size,
        power=0.8,
        alpha=0.05,
        outlier_percentile=99.5,
    )

    experiment_effect_sizes = res._params["simulated_values"]
    cols_no_stats = set(list(experiment_effect_sizes))
    stats_cols = {
        "mean",
        "std",
        "mean_trimmed",
        "std_trimmed",
        "trim_change_mean",
        "trim_change_std",
    }
    cols_with_stats = cols_no_stats | stats_cols

    with pytest.raises(ValueError):
        _ = res.pretty_results(append_stats=True)

    no_stats = res.pretty_results()
    assert set(no_stats.data.columns) == cols_no_stats

    also_no_stats = res.pretty_results(input_data=df, append_stats=False)
    assert set(also_no_stats.data.columns) == cols_no_stats

    with_stats = res.pretty_results(input_data=df, append_stats=True)
    assert set(with_stats.data.columns) == cols_with_stats

    # make sure highlight_listthan won't throw an error
    _ = res.pretty_results(
        input_data=df,
        append_stats=True,
        highlight_lessthan=[(10, "green"), (20, "blue")],
    )

    # check that subset works
    subset_experiment_effect_sizes = experiment_effect_sizes[1:-1]
    subset_cols_with_stats = set(subset_experiment_effect_sizes) | stats_cols

    no_stats = res.pretty_results(simulated_values=subset_experiment_effect_sizes)
    assert set(no_stats.data.columns) == set(subset_experiment_effect_sizes)

    with_stats = res.pretty_results(
        input_data=df,
        append_stats=True,
        simulated_values=subset_experiment_effect_sizes,
    )
    assert set(with_stats.data.columns) == subset_cols_with_stats


def test_results_holder():
    df = pd.DataFrame(
        {
            search_clients_daily.name: np.random.normal(size=100),
            uri_count.name: np.random.normal(size=100),
        }
    )

    metrics = [search_clients_daily, uri_count]
    res = z_or_t_ind_sample_size_calc(df, metrics)
    metric_names = set([el.name for el in metrics])

    # set should only have 2 values: "search_clients_daily" "uri_count"
    assert len(res) == len(metric_names)
    assert set(res.keys()) == metric_names
    assert len(res.values()) == len(metric_names)
    iter_keys = []
    for key, _ in res.items():
        iter_keys.append(key)
    assert len(iter_keys) == len(metric_names)
    assert set(iter_keys) == metric_names

    # make sure the  get_dataframe isn't broken
    _ = res.get_dataframe()
    res.plot_results()
