import pytest
from cheap_lint import sql_lint

import mozanalysis.metrics.desktop as mmd
import mozanalysis.segments.desktop as msd
from mozanalysis.segments import Segment, SegmentDataSource

from . import enumerate_included


@pytest.fixture()
def included_segments():
    return enumerate_included((msd,), Segment)


@pytest.fixture()
def included_segment_datasources():
    return enumerate_included((msd,), SegmentDataSource)


def test_imported_ok():
    assert msd.regular_users_v3


def test_sql_not_detectably_malformed(included_segments, included_segment_datasources):
    for _name, s in included_segments:
        sql_lint(s.select_expr)

    for _name, sds in included_segment_datasources:
        sql_lint(sds.from_expr_for(None))


def test_consistency_of_segment_and_variable_names(included_segments):
    for name, segment in included_segments:
        assert name == segment.name, segment


def test_segment_data_source_window_end_validates():
    msd.SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=0,
        window_end=0,
    )

    with pytest.raises(ValueError):
        msd.SegmentDataSource(
            name="bla",
            from_expr="bla",
            window_start=0,
            window_end=1,
        )


def test_segment_data_source_window_start_validates():
    msd.SegmentDataSource(
        name="bla",
        from_expr="bla",
        window_start=-1,
        window_end=-1,
    )

    with pytest.raises(ValueError):
        msd.SegmentDataSource(
            name="bla",
            from_expr="bla",
            window_start=0,
            window_end=-1,
        )


def test_segment_validates_not_metric_data_source():
    with pytest.raises(TypeError):
        msd.Segment(
            name="bla",
            data_source=mmd.clients_daily,
            select_expr="bla",
        )


def test_included_segments_have_docs(included_segments):
    for name, segment in included_segments:
        assert segment.friendly_name and segment.description, name


def test_complains_about_template_without_default():
    with pytest.raises(ValueError):
        msd.SegmentDataSource(
            name="foo",
            from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        )
    msd.SegmentDataSource(
        name="foo",
        from_expr="moz-fx-data-shared-prod.{dataset}.foo",
        default_dataset="dataset",
    )
