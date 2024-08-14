from pathlib import Path

import bioviz_accessor  # noqa: F401
import joblib
import pytest
import xarray as xr

data = xr.load_dataarray(Path(__file__).parent / "test_data.zarr", engine="zarr")


def test_underspecified_coords():
    with pytest.raises(ValueError):
        data.bviz.stitched()


def test_output_consistency():
    # check that we have exactly the same stitched data
    assert (
        joblib.hash(data.bviz.stitched(T=0, C=0, Z=0).data)
        == "8c29d6eba80a4e978ec987ee6797586b"
    )

    # this should be different!
    assert (
        joblib.hash(data.bviz.stitched(T=1, C=0, Z=0).data)
        != "82edf77404fcaee27b750c6cbbeae0c0"
    )


def test_cache():
    assert len(data.bviz._stitched_cache) == 2
    data.bviz.max_cache = 2
    data.bviz.stitched(T=1, C=1, Z=0)
    # should still be length 2
    assert len(data.bviz._stitched_cache) == 2
    data.bviz.max_cache = 1

    # one item should have been removed
    assert len(data.bviz._stitched_cache) == 1


def test_smoke_hyperslicer():
    # this is surprisingly difficult to test
    # rely on `mpl-interactions` tests
    # and just check that nothing breaks
    data.bviz.hypersliced()
