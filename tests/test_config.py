# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest

from cheap_lint import sql_lint
from mozanalysis.config import ConfigLoader
from mozanalysis.metrics import DataSource, Metric
from mozanalysis.segments import Segment, SegmentDataSource


def test_get_metric():
    m = ConfigLoader.get_metric("active_hours", "firefox_desktop")
    assert isinstance(m, Metric)
    assert m.name == "active_hours"
    assert m.friendly_name
    assert m.description
    assert m.data_source
    sql_lint(m.select_expr)


def test_unknown_metric_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_metric("fake_metric", "firefox_desktop")


def test_get_data_source():
    d = ConfigLoader.get_data_source("clients_daily", "firefox_desktop")
    assert isinstance(d, DataSource)
    assert d.name == "clients_daily"
    assert d.client_id_column == "client_id"
    assert d.submission_date_column == "submission_date"
    sql_lint(d.from_expr_for("dataset"))


def test_unknown_data_source_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_data_source("fake_data_source", "firefox_desktop")


def test_get_segment():
    s = ConfigLoader.get_segment("suggest", "firefox_desktop")
    assert isinstance(s, Segment)
    assert s.name == "suggest"
    assert s.data_source
    sql_lint(s.select_expr)


def test_unknown_segment_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_segment("fake_segment", "firefox_desktop")


def test_get_segment_data_source():
    s = ConfigLoader.get_segment_data_source("clients_last_seen", "firefox_desktop")
    assert isinstance(s, SegmentDataSource)
    assert s.name == "clients_last_seen"
    assert s.client_id_column == "client_id"
    assert s.submission_date_column == "submission_date"
    assert s.window_start == 0
    assert s.window_end == 0
    sql_lint(s.from_expr_for("dataset"))


def test_unknown_segment_data_source_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_segment_data_source(
            "fake_segment_data_source", "firefox_desktop"
        )


def test_get_outcome_metric_outcome_data_source():
    m = ConfigLoader.get_outcome_metric(
        "cert_error_page_loaded", "networking", "firefox_desktop"
    )
    assert isinstance(m, Metric)
    assert m.name == "cert_error_page_loaded"
    assert m.friendly_name
    assert m.data_source
    sql_lint(m.select_expr)


def test_get_outcome_metric_general_data_source():
    m = ConfigLoader.get_outcome_metric(
        "urlbar_search_count", "firefox_suggest", "firefox_desktop"
    )
    assert isinstance(m, Metric)
    assert m.name == "urlbar_search_count"
    assert m.friendly_name
    assert m.description
    assert m.data_source
    sql_lint(m.select_expr)


def test_get_parametrized_outcome_metric_fails():
    m = ConfigLoader.get_outcome_metric(
        "spotlight_impressions", "spotlight_engagement", "firefox_desktop"
    )
    assert isinstance(m, Metric)
    assert m.name == "spotlight_impressions"
    assert m.friendly_name
    assert m.description
    assert m.data_source
    sql_lint(m.select_expr)


def test_outcome_unknown_metric_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_outcome_metric("fake_metric", "networking", "firefox_desktop")


def test_unknown_outcome_metric_fails():
    with pytest.raises(Exception):
        ConfigLoader.get_outcome_metric(
            "fake_metric", "fake_outcome", "firefox_desktop"
        )


def test_get_outcome_data_source():
    d = ConfigLoader.get_outcome_data_source(
        "events_certerror", "networking", "firefox_desktop"
    )
    assert isinstance(d, DataSource)
    assert d.name == "events_certerror"
    assert d.submission_date_column == "submission_date"
    sql_lint(d.from_expr_for("dataset"))


def test_outcome_unknown_data_source():
    with pytest.raises(Exception):
        ConfigLoader.get_outcome_data_source(
            "fake_data_source", "networking", "firefox_desktop"
        )


def test_unknown_outcome_data_source():
    with pytest.raises(Exception):
        ConfigLoader.get_outcome_data_source(
            "fake_data_source", "fake_outcome", "firefox_desktop"
        )
