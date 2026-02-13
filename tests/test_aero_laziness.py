import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.utils import Grid, _get_grid_type, create_regridding_dataset, format_for_regrid


def test_get_grid_type_laziness():
    """Verify that _get_grid_type does not compute large 2D coordinates."""

    def raise_if_computed(x):
        # Dask map_blocks might call this with a small sample for meta inference
        # unless we specify chunks and dtype.
        if x.size > 1:
            msg = "Data was computed!"
            raise RuntimeError(msg)
        return x

    # Create 2D coordinates that will raise error if computed
    # Use a large enough array and specify chunks/dtype
    lat_data = da.from_array(np.random.rand(100, 100), chunks=(50, 50)).map_blocks(raise_if_computed, dtype=float)
    lon_data = da.from_array(np.random.rand(100, 100), chunks=(50, 50)).map_blocks(raise_if_computed, dtype=float)

    ds = xr.Dataset(coords={"latitude": (("y", "x"), lat_data), "longitude": (("y", "x"), lon_data)})

    # _get_grid_type should return CURVILINEAR without computing the 2D data
    try:
        grid_type = _get_grid_type(ds)
        assert grid_type.name == "CURVILINEAR"
    except RuntimeError:
        pytest.fail("_get_grid_type triggered eager computation of 2D coordinates!")


def test_create_regridding_dataset_chunks():
    """Verify that create_regridding_dataset respects the chunks parameter."""
    grid = Grid(north=90, south=-90, east=180, west=-180, resolution_lat=1, resolution_lon=1)

    # Xarray dimension coordinates are often computed upon Dataset creation
    # to create indexes. To check if our function is using dask correctly,
    # we can check the behavior of create_lat_lon_coords directly or
    # check the data property if it's still dask-backed.
    from monet_regrid.utils import create_lat_lon_coords

    # Default chunks
    lat_default, _ = create_lat_lon_coords(grid)
    assert lat_default.chunks is not None

    # Custom chunks
    lat_chunked, _ = create_lat_lon_coords(grid, chunks=10)
    assert lat_chunked.chunks[0][0] == 10


def test_format_for_regrid_laziness():
    """Verify that format_for_regrid maintains laziness for data variables."""
    # Data variables should definitely stay lazy

    lats = np.arange(-80, 81, 10)
    lons = np.arange(-170, 171, 20)

    # Use a large-ish data array
    data = da.random.random((len(lats), len(lons)), chunks=(5, 5))

    ds = xr.Dataset(data_vars={"test": (("lat", "lon"), data)}, coords={"lat": lats, "lon": lons})

    target = create_regridding_dataset(Grid(north=90, south=-90, east=180, west=-180, resolution_lat=5, resolution_lon=5))

    formatted = format_for_regrid(ds, target)

    # The data variable should still be a Dask array
    assert isinstance(formatted["test"].data, da.Array)
    # Check that it's still lazy (not computed)
    assert formatted["test"].data.npartitions > 0
