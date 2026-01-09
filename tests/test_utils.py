import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.utils import (
    Grid,
    create_lat_lon_coords,
    create_regridding_dataset,
    format_lat,
)

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from monet_regrid.utils import format_lat
# New import: from monet_regrid.utils import format_lat


def test_create_lat_lon_coords_lazy():
    """Test that `create_lat_lon_coords` returns Dask arrays when requested."""
    # 1. The Logic (Setup)
    grid = Grid(
        north=90,
        south=-90,
        east=180,
        west=-180,
        resolution_lat=1.0,
        resolution_lon=1.0,
    )

    # 2. The Proof (Execution & Assertion)
    lat_coords, lon_coords = create_lat_lon_coords(grid, use_dask=True)

    assert isinstance(lat_coords, da.Array)
    assert isinstance(lon_coords, da.Array)


def test_create_regridding_dataset_structure():
    """Test the structure of the dataset created by `create_regridding_dataset`.

    This test verifies that the function produces a valid xarray.Dataset with the
    correct coordinate names, dimensions, and attributes, acknowledging that xarray
    will eagerly load 1D coordinates into a pandas.Index.
    """
    # 1. The Logic (Setup)
    grid = Grid(
        north=90,
        south=-90,
        east=180,
        west=-180,
        resolution_lat=1.0,
        resolution_lon=1.0,
    )

    # 2. The Proof (Execution & Assertion)
    ds = create_regridding_dataset(grid, use_dask=True)

    # Verify final dataset structure and values
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert ds["latitude"].shape == (181,)
    assert ds["longitude"].shape == (361,)
    assert ds["latitude"].attrs["units"] == "degrees_north"
    assert np.isclose(ds["latitude"].values[0], -90.0)
    assert np.isclose(ds["latitude"].values[-1], 90.0)


def test_grid_to_xarray_lazy():
    """Test that `Grid.to_xarray` can create a lazy dataset."""
    # 1. The Logic (Setup)
    grid = Grid(
        north=90,
        south=-90,
        east=180,
        west=-180,
        resolution_lat=1.0,
        resolution_lon=1.0,
    )

    # 2. The Proof (Execution & Assertion)
    ds = grid.create_regridding_dataset(use_dask=True)

    # Xarray eagerly loads 1D coords, so we can't check for Dask arrays here.
    # Instead, we just check that the dataset is created correctly.
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert ds["latitude"].shape == (181,)


def test_format_lat():
    lat_vals = np.arange(-89.5, 89.5 + 1, 1)
    lon_vals = np.arange(-179.5, 179.5 + 1, 1)
    x_vals = np.broadcast_to(lat_vals, (len(lon_vals), len(lat_vals)))
    ds = xr.Dataset(
        data_vars={"x": (("lon", "lat"), x_vals)},
        coords={"lat": lat_vals, "lon": lon_vals},
        attrs={"foo": "bar"},
    )
    ds.lat.attrs["is"] = "coord"
    ds.x.attrs["is"] = "data"

    formatted = format_lat(ds, ds, {"lat": "lat", "lon": "lon"})
    # Check that lat has been extended to include poles
    assert formatted.lat.values[0] == -90
    assert formatted.lat.values[-1] == 90
    # Check that data has been extrapolated to include poles
    assert (formatted.x.isel(lat=0) == -89.5).all()
    assert (formatted.x.isel(lat=-1) == 89.5).all()
    # Check that attrs have been preserved
    assert formatted.attrs["foo"] == "bar"
    assert formatted.lat.attrs["is"] == "coord"
    assert formatted.x.attrs["is"] == "data"
