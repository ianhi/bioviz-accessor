from pathlib import Path

import bioviz_accessor  # noqa: F401
import pytest
import xarray as xr
from dask.base import tokenize

data = xr.load_dataarray(Path(__file__).parent / "test_data.zarr", engine="zarr")


def test_underspecified_coords():
    with pytest.raises(ValueError):
        data.bviz.stitched()


def test_output_consistency():
    # check that we have exactly the same stitched data
    assert (
        tokenize(data.bviz.stitched(T=0, C=0, Z=0).data, ensure_deterministic=True)
        == "82edf77404fcaee27b750c6cbbeae0c0"
    )
    print(data.bviz._stitched_cache)

    # this should be different!
    assert (
        tokenize(data.bviz.stitched(T=1, C=0, Z=0).data, ensure_deterministic=True)
        != "82edf77404fcaee27b750c6cbbeae0c0"
    )
    print(data.bviz._stitched_cache)


def test_cache():
    print(data.bviz._stitched_cache)
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


# holding off this until an update in mda-simulator
# that will allow us to set the rng seed
# def test_generate_data():
# todo...
# call gen data
# check the tokenization
