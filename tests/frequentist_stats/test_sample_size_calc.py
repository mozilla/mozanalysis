# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from dataclasses import dataclass
import datetime

import numpy as np
import pandas as pd
import pytest

from mozanalysis.frequentist_stats import sample_size
from mozanalysis.frequentist_stats.sample_size import (
    z_or_t_ind_sample_size_calc,
    empirical_effect_size_sample_size_calc,
    get_firefox_release_dates,
    SampleSizing,
)
from mozanalysis.metrics.desktop import search_clients_daily, uri_count
from mozanalysis.segments.desktop import regular_users_v3


@pytest.fixture
def mock_today(monkeypatch):
    class MockDate(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date(2023, 12, 31)

    monkeypatch.setattr(datetime, "date", MockDate)


@pytest.fixture
def mock_firefox_release_dates(monkeypatch):
    def _mock_release_dates():
        return {
            99: "2023-11-08",
            100: "2023-11-25",
            101: "2023-12-07",
            102: "2023-12-22",
        }

    monkeypatch.setattr(sample_size, "get_firefox_release_dates", _mock_release_dates)


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


def test_get_firefox_release_dates():
    # Pulls real data from Firefox Release Calendar website
    result = get_firefox_release_dates()
    assert result[3] == "2008-06-17"
    assert result[120] == "2023-11-21"


def test_sample_sizing_init(mock_today, mock_firefox_release_dates, capsys):
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


def test_sample_sizing_no_end_date(mock_today, mock_firefox_release_dates):
    pass


def test_sample_sizing_late_end_date(mock_today, mock_firefox_release_dates):
    pass


def test_sample_sizing_no_sampling(mock_today, mock_firefox_release_dates):
    pass
