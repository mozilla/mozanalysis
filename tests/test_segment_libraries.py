import pytest
from cheap_lint import sql_lint
from mozanalysis.config import ConfigLoader
from mozanalysis.segments import Segment, SegmentDataSource


@pytest.fixture()
def included_segments():
    desktop_segments = [
        ConfigLoader.get_segment("regular_users_v3", "firefox_desktop"),
        ConfigLoader.get_segment("new_or_resurrected_v3", "firefox_desktop"),
        ConfigLoader.get_segment("weekday_regular_v1", "firefox_desktop"),
        ConfigLoader.get_segment("allweek_regular_v1", "firefox_desktop"),
        ConfigLoader.get_segment("new_unique_profiles", "firefox_desktop"),
    ]
    return desktop_segments


@pytest.fixture()
def included_segment_datasources():
    return [
        ConfigLoader.get_segment_data_source("clients_last_seen", "firefox_desktop")
    ]


def test_sql_not_detectably_malformed(included_segments, included_segment_datasources):
    for s in included_segments:
        sql_lint(s.select_expr)

    for sds in included_segment_datasources:
        sql_lint(sds.from_expr_for(None))


def test_segment_data_source_window_end_validates():
    SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=0,
        window_end=0,
    )

    SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=0,
        window_end=1,
    )

    SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=1,
        window_end=3,
    )


def test_segment_data_source_window_start_validates():
    SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=-1,
        window_end=-1,
    )

    with pytest.raises(ValueError, match="window_start must be <= window_end"):
        SegmentDataSource(
            name="bla",
            from_expr="bla",
            window_start=0,
            window_end=-1,
        )


def test_segment_validates_not_metric_data_source():
    with pytest.raises(TypeError):
        Segment(
            name="bla",
            data_source=ConfigLoader.get_data_source(
                "clients_daily", "firefox_desktop"
            ),
            select_expr="bla",
        )


def test_included_segments_have_docs(included_segments):
    for segment in included_segments:
        assert segment.friendly_name
        assert segment.description


def test_complains_about_template_without_default():
    with pytest.raises(
        ValueError,
        match="foo: from_expr contains a dataset template but no value was provided.",
    ):
        SegmentDataSource(
            name="foo",
            from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        )
    SegmentDataSource(
        name="foo",
        from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        default_dataset="dataset",
    )
