import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.utils import _create_lat_lon_from_dims, format_lat


def test_format_lat():
    """Test the format_lat function for pole padding."""
    # Create a sample DataArray that is global but doesn't include the poles
    lat = np.arange(-89.5, 90.5, 1)
    lon = np.arange(0, 360, 1)
    data = np.random.rand(len(lat), len(lon))
    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )

    # The target grid is not used in the current implementation of format_lat
    target_ds = xr.Dataset()
    formatted_coords = {"lat": "lat", "lon": "lon"}

    # Apply the formatting
    formatted_da = format_lat(da, target_ds, formatted_coords)

    # Check that the poles have been added
    assert formatted_da.lat.min() == -90
    assert formatted_da.lat.max() == 90
    assert len(formatted_da.lat) == len(lat) + 2

    # Check that the pole values are the mean of the original boundary latitudes
    np.testing.assert_allclose(formatted_da.isel(lat=0).values, da.isel(lat=0).mean("lon").values)
    np.testing.assert_allclose(formatted_da.isel(lat=-1).values, da.isel(lat=-1).mean("lon").values)


def test_create_lat_lon_from_dims():
    """Test lazy coordinate generation for a Dask-backed array."""
    # Create a sample Dask-backed DataArray
    data = da.random.random((10, 20), chunks=(5, 10))
    source_da = xr.DataArray(data, dims=["y", "x"])

    # Generate the coordinate dataset
    coord_ds = _create_lat_lon_from_dims(source_da)

    # --- Validation ---
    # 1. Check that the coordinates are Dask-backed (lazy)
    assert hasattr(coord_ds["latitude"].data, "dask")
    assert hasattr(coord_ds["longitude"].data, "dask")

    # 2. Check shapes and dimensions
    assert coord_ds["latitude"].shape == (10, 20)
    assert coord_ds["longitude"].shape == (10, 20)
    assert coord_ds["latitude"].dims == ("y", "x")
    assert coord_ds["longitude"].dims == ("y", "x")

    # 3. Compute the result and check values
    computed_ds = coord_ds.compute()
    expected_lon, expected_lat = np.meshgrid(np.arange(20), np.arange(10))
    np.testing.assert_allclose(computed_ds["latitude"].values, expected_lat)
    np.testing.assert_allclose(computed_ds["longitude"].values, expected_lon)


def test_create_lat_lon_from_dims_numpy():
    """Test coordinate generation for a NumPy-backed array."""
    # Create a sample NumPy-backed DataArray
    data = np.random.rand(10, 20)
    source_da = xr.DataArray(data, dims=["y", "x"])

    # Generate the coordinate dataset
    coord_ds = _create_lat_lon_from_dims(source_da)

    # --- Validation ---
    # 1. Check that the coordinates are NumPy-backed
    assert isinstance(coord_ds["latitude"].data, np.ndarray)
    assert isinstance(coord_ds["longitude"].data, np.ndarray)

    # 2. Check shapes and dimensions
    assert coord_ds["latitude"].shape == (10, 20)
    assert coord_ds["longitude"].shape == (10, 20)

    # 3. Check values
    expected_lon, expected_lat = np.meshgrid(np.arange(20), np.arange(10))
    np.testing.assert_allclose(coord_ds["latitude"].values, expected_lat)
    np.testing.assert_allclose(coord_ds["longitude"].values, expected_lon)


def test_create_lat_lon_from_dims_invalid_input():
    """Test that the function raises an error for invalid input."""
    # Create a DataArray with fewer than two dimensions
    source_da = xr.DataArray(np.random.rand(10), dims=["y"])

    with pytest.raises(ValueError, match="Source data must have at least 2 dimensions"):
        _create_lat_lon_from_dims(source_da)
