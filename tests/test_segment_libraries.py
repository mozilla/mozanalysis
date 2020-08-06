import pytest
from cheap_lint import sql_lint

import mozanalysis.segments.desktop as msd
import mozanalysis.metrics.desktop as mmd


def test_imported_ok():
    assert msd.regular_users_v3


def test_sql_not_detectably_malformed():
    for m in msd.__dict__.values():
        if isinstance(m, msd.Segment):
            sql_lint(m.select_expr)

    for ds in msd.__dict__.values():
        if isinstance(ds, msd.SegmentDataSource):
            sql_lint(ds.from_expr)


def test_consistency_of_segment_and_variable_names():
    for name, segment in msd.__dict__.items():
        if isinstance(segment, msd.Segment):
            assert name == segment.name, segment


def test_segment_data_source_window_end_validates():
    msd.SegmentDataSource(
        name='bla',
        from_expr="bla",
        window_start=0,
        window_end=0,
    )

    with pytest.raises(ValueError):
        msd.SegmentDataSource(
            name='bla',
            from_expr="bla",
            window_start=0,
            window_end=1,
        )


def test_segment_data_source_window_start_validates():
    msd.SegmentDataSource(
        name='bla',
        from_expr="bla",
        window_start=-1,
        window_end=-1,
    )

    with pytest.raises(ValueError):
        msd.SegmentDataSource(
            name='bla',
            from_expr="bla",
            window_start=0,
            window_end=-1,
        )


def test_segment_validates_not_metric_data_source():
    with pytest.raises(TypeError):
        msd.Segment(
            name='bla',
            data_source=mmd.clients_daily,
            select_expr='bla',
        )
