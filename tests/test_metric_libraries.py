import mozanalysis.metrics.desktop as mmd
import mozanalysis.metrics.firetv as mmfftv
import mozanalysis.metrics.fenix as mmf


def test_imported_ok():
    assert mmd.active_hours
    assert mmfftv.home_tile_clicks
    assert mmf.duration
