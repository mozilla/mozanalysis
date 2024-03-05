# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from dataclasses import dataclass
import datetime

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import pytest

from mozanalysis.frequentist_stats import sample_size
from mozanalysis.frequentist_stats.sample_size import (
    z_or_t_ind_sample_size_calc,
    empirical_effect_size_sample_size_calc,
    get_firefox_release_dates,
    SampleSizing,
)
from mozanalysis.metrics.desktop import search_clients_daily, uri_count, active_hours
from mozanalysis.segments.desktop import regular_users_v3


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


@pytest.fixture
def mock_historical_target(fake_ts_result, monkeypatch):
    class MockHistoricalTarget:
        def __init__(self, **kwargs):
            pass

        def get_time_series_data(
            self,
            bq_context,
            metric_list,
            time_series_period,
            target_list,
            **kwargs,
        ):
            return fake_ts_result

        def get_single_window_data(
            self,
            bq_context,
            metric_list,
            target_list,
            **kwargs,
        ):
            fake_vals = np.arange(10)
            df = pd.DataFrame({"client_id": [f"c{i}" for i in range(len(fake_vals))]})
            for m in metric_list:
                df[m.name] = fake_vals
            return df

    monkeypatch.setattr(sample_size, "HistoricalTarget", MockHistoricalTarget)


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
    result = empirical_effect_size_sample_size_calc(
        res=fake_ts_result, bq_context=None, metric_list=[active_hours, uri_count]
    )

    assert len(result) == 2
    for m in ["active_hours", "uri_count"]:
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


def test_get_firefox_release_dates():
    # Pulls real data from Firefox Release Calendar website
    result = get_firefox_release_dates()
    assert result[3] == "2008-06-17"
    assert result[120] == "2023-11-21"


class TestSampleSizing:
    @pytest.fixture(autouse=True)
    def mock_today(self, monkeypatch):
        class MockDate(datetime.date):
            @classmethod
            def today(cls):
                return datetime.date(2023, 12, 31)

        monkeypatch.setattr(sample_size, "date", MockDate)

    @pytest.fixture(autouse=True)
    def mock_firefox_release_dates(self, monkeypatch):
        def _mock_release_dates():
            return {
                99: "2023-11-08",
                100: "2023-11-25",
                101: "2023-12-07",
                102: "2023-12-22",
            }

        monkeypatch.setattr(
            sample_size, "get_firefox_release_dates", _mock_release_dates
        )

    @pytest.fixture
    def sample_sizer(self, mock_historical_target):
        # Default instance to use for testing sample size calculations
        return SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count, active_hours],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date="2023-12-15",
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0.05,
        )

    @pytest.fixture
    def sample_sizer_no_sampling(self, mock_historical_target):
        # Instance to use for testing sample size calculations with no sampling
        return SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count, active_hours],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date="2023-12-15",
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0,
        )

    def test_init(self, capsys):
        # "today" is 2023-12-31
        s = SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date="2023-12-15",
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0.05,
        )

        assert s.bq_context == "bqcontext"
        assert s.experiment_name == "test"
        assert s.metrics == [uri_count]
        assert s.n_days_observation_max == 7
        assert s.n_days_enrollment == 3
        assert s.n_days_launch_after_version_release == 5
        assert s.end_date == "2023-12-15"
        assert s.alpha == 0.02
        assert s.power == 0.8
        assert s.outlier_percentile == 0.99
        assert s.sample_rate == 0.05
        assert s.total_historical_period_days == 15
        assert s.min_major_version == 100
        assert s.min_major_version_date == "2023-11-25"
        assert s.enrollment_start_date == "2023-11-30"

        assert len(s.targets) == 2
        assert s.targets[0] == regular_users_v3
        t = s.targets[1]
        assert "browser_version_info.major_version >= 100" in t.select_expr
        assert "sample_id < 5" in t.select_expr
        assert t.data_source.name == "clients_daily"

        # Check text printed to stdout
        captured = capsys.readouterr()
        assert not captured.err
        printed = captured.out.split("\n\n")

        assert "5 days wait after new version" in printed[0]
        assert "3 days enrollment" in printed[0]
        assert "7 days observation" in printed[0]
        assert "version 100 released on 2023-11-25" in printed[0]
        assert "start on 2023-11-30" in printed[0]
        assert "sampled at a rate of 5%" in printed[0]

        target_str = printed[1].split("\n")
        assert "is_regular_user_v3" in target_str[1]
        assert (
            "(browser_version_info.major_version >= 100) AND (sample_id < 5)"
            in target_str[2]
        )

        assert s.population_size is None
        printed = capsys.readouterr().out
        assert "size will be set" in printed

    def test_no_end_date(self, capsys):
        # "today" is 2023-12-31
        s = SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date=None,
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0.05,
        )

        assert s.end_date == None
        assert s.total_historical_period_days == 15
        assert s.min_major_version == 101
        assert s.min_major_version_date == "2023-12-07"
        assert s.enrollment_start_date == "2023-12-12"

        # Check text printed to stdout
        captured = capsys.readouterr()
        printed = captured.out.split("\n\n")

        assert "version 101 released on 2023-12-07" in printed[0]
        assert "start on 2023-12-12" in printed[0]

    def test_late_end_date(self, capsys):
        # "today" is 2023-12-31
        s = SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date="2024-01-05",
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0.05,
        )

        assert s.end_date == "2024-01-05"
        assert s.min_major_version == 101
        assert s.min_major_version_date == "2023-12-07"
        assert s.enrollment_start_date == "2023-12-12"

        # Check text printed to stdout
        captured = capsys.readouterr()
        assert not captured.err
        printed = captured.out.split("\n\n")

        assert "end date 2024-01-05" in printed[0]
        assert "latest feasible date 2023-12-29" in printed[0]
        assert "Using 2023-12-29" in printed[0]

        assert "version 101 released on 2023-12-07" in printed[1]
        assert "start on 2023-12-12" in printed[1]

    def test_no_sampling(self, capsys):
        # "today" is 2023-12-31
        s = SampleSizing(
            experiment_name="test",
            bq_context="bqcontext",
            metrics=[uri_count],
            targets=[regular_users_v3],
            n_days_observation_max=7,
            n_days_enrollment=3,
            n_days_launch_after_version_release=5,
            end_date="2023-12-15",
            alpha=0.02,
            power=0.8,
            outlier_percentile=0.99,
            sample_rate=0,
        )

        assert s.sample_rate == 0

        assert len(s.targets) == 2
        assert s.targets[0] == regular_users_v3
        t = s.targets[1]
        assert "browser_version_info.major_version >= 100" in t.select_expr
        assert "sample_id" not in t.select_expr

        # Check text printed to stdout
        captured = capsys.readouterr()
        assert not captured.err
        printed = captured.out.split("\n\n")

        assert "sampled at a rate" not in printed[0]

        target_str = printed[1].split("\n")
        assert "(browser_version_info.major_version >= 100)" in target_str[2]
        assert "sample_id" not in target_str[2]

    def test_sample_sizes(self, sample_sizer, capsys):
        sizing_df, stats_df = sample_sizer.sample_sizes([0.02, 0.03], [3, 5])

        assert isinstance(sizing_df, pd.DataFrame)
        assert sizing_df.index.nlevels == 3
        assert sizing_df.index.get_level_values(0).to_list() == [3, 3, 3, 3, 5, 5, 5, 5]
        assert (
            sizing_df.index.get_level_values(1).to_list()
            == [0.02, 0.02, 0.03, 0.03] * 2
        )
        assert (
            sizing_df.index.get_level_values(2).to_list()
            == ["active_hours", "uri_count"] * 4
        )
        np.testing.assert_allclose(
            sizing_df.iloc[0]["sample_size_per_branch"], 23522, atol=0.5
        )
        np.testing.assert_allclose(
            sizing_df.iloc[0]["population_percent_per_branch"], 11761, atol=0.5
        )
        np.testing.assert_allclose(
            sizing_df.iloc[2]["sample_size_per_branch"], 10454, atol=0.5
        )
        np.testing.assert_allclose(
            sizing_df.iloc[2]["population_percent_per_branch"], 5227, atol=0.5
        )

        assert isinstance(stats_df, pd.DataFrame)
        assert stats_df.index.nlevels == 2
        assert stats_df.index.get_level_values(0).to_list() == [3, 3, 5, 5]
        assert (
            stats_df.index.get_level_values(1).to_list()
            == ["active_hours", "uri_count"] * 2
        )
        assert stats_df.iloc[0]["mean"] == 4.5
        assert stats_df.iloc[0]["mean_trimmed"] == 4.0
        np.testing.assert_allclose(stats_df.iloc[0]["std"], 3.027, atol=0.001)
        np.testing.assert_allclose(stats_df.iloc[0]["std_trimmed"], 2.739, atol=0.001)

        printed = capsys.readouterr().out.strip().split("\n")
        assert "3 days enrollment" in printed[0]
        assert "3 days observation" in printed[0]

        assert "population size: 200" in printed[1]
        assert "rate of 5%" in printed[1]

        assert "3 days enrollment" in printed[-1]
        assert "5 days observation" in printed[-1]

        assert sample_sizer.population_size == 200
        # Retrieving population size shouldn't generate any printed message
        assert not capsys.readouterr().out

    def test_sample_sizes_no_obs(self, sample_sizer):
        sizing_df, stats_df = sample_sizer.sample_sizes([0.02, 0.03])

        # Just check that the right number of results were returned
        # and n_days_observation was updated
        assert sizing_df.index.get_level_values(0).to_list() == [7, 7, 7, 7]
        assert stats_df.index.get_level_values(0).to_list() == [7, 7]

    def test_sample_sizes_bad_obs(self, sample_sizer):
        with pytest.raises(ValueError):
            sample_sizer.sample_sizes([0.02, 0.03], [7, 14])

    def test_sample_sizes_no_sampling(self, sample_sizer_no_sampling, capsys):
        sizing_df, stats_df = sample_sizer_no_sampling.sample_sizes([0.02, 0.03], [3])

        np.testing.assert_allclose(
            sizing_df.iloc[0]["population_percent_per_branch"], 235219, atol=0.5
        )
        np.testing.assert_allclose(
            sizing_df.iloc[2]["population_percent_per_branch"], 104542, atol=0.5
        )

        printed = capsys.readouterr().out.strip().split("\n")
        assert "population size: 10" in printed[1]
        assert "sampled at a rate" not in printed[1]

        assert sample_sizer_no_sampling.population_size == 10

    def test_display_sample_size_results(self, sample_sizer, capsys):
        sizing_df, stats_df = sample_sizer.sample_sizes([0.02, 0.03], [3, 5])
        # Clear buffer
        capsys.readouterr()

        s = sample_sizer.display_sample_size_results(
            sizing_result=sizing_df,
            n_days_observation=3,
            stats_df=stats_df,
            show_population_pct=True,
            rel_effect_sizes=None,
        )

        assert isinstance(s, Styler)

        printed = capsys.readouterr().out
        assert "3 days observation" in printed
        assert "population size: 200" in printed
        assert "rate of 5%" in printed

    def test_empirical_sizing(self, sample_sizer, capsys):
        df = sample_sizer.empirical_sizing()

        assert isinstance(df, pd.DataFrame)
        assert df.index.to_list() == ["uri_count", "active_hours"]
        expected = {
            "relative_effect_size": 0.6,
            "effect_size_value": 3,
            "mean_value": 5,
            "std_dev_value": 2,
            "effect_size_period": 7,
            "mean_period": 0,
            "std_dev_period": 7,
        }
        for k, v in expected.items():
            assert df.iloc[0][k] == v
        np.testing.assert_allclose(df.iloc[0]["sample_size_per_branch"], 11, atol=0.5)
        np.testing.assert_allclose(
            df.iloc[0]["population_percent_per_branch"], 0.054, atol=0.001
        )

        printed = capsys.readouterr().out.strip().split("\n")
        assert "empirical sizing" in printed[0]
        assert "3 days enrollment" in printed[0]
        assert "7 days observation" in printed[0]

        assert "population size: 20,000" in printed[1]
        assert "rate of 5%" in printed[1]

        assert sample_sizer.population_size == 20000
        # Retrieving population size shouldn't generate any printed message
        assert not capsys.readouterr().out

    def test_empirical_sizing_no_sampling(self, sample_sizer_no_sampling, capsys):
        df = sample_sizer_no_sampling.empirical_sizing()

        np.testing.assert_allclose(
            df.iloc[0]["population_percent_per_branch"], 1.09, atol=0.005
        )

        printed = capsys.readouterr().out.strip().split("\n")
        assert "population size: 1,000" in printed[1]
        assert "sampled at a rate" not in printed[1]

        assert sample_sizer_no_sampling.population_size == 1000

    def test_empirical_sizing_with_styler(self, sample_sizer):
        sdf = sample_sizer.empirical_sizing(style=True)
        assert isinstance(sdf, Styler)

    def test_style_empirical_sizing(self, sample_sizer, capsys):
        df = sample_sizer.empirical_sizing()
        # Clear buffer
        capsys.readouterr()

        styled = sample_sizer.style_empirical_sizing_result(df)
        assert isinstance(styled, Styler)

        assert styled.data.index.to_list() == ["uri_count", "active_hours"]
        assert styled.data.shape[1] == 6

        printed = capsys.readouterr().out
        assert "population size: 20,000" in printed
        assert "rate of 5%" in printed
