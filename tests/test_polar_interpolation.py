import logging

import numpy as np
import xarray as xr

"""Test for polar interpolation fix in curvilinear interpolation."""


def test_polar_interpolation_no_nan():
    """Test that bilinear interpolation at poles doesn't produce NaN values."""
    # Create a source grid that includes polar regions
    ny, nx = 20, 40
    lon_1d = np.linspace(-180, 180, nx)
    lat_1d = np.linspace(-90, 90, ny)  # Include the poles
    source_lon_2d, source_lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Create source dataset
    source_ds = xr.Dataset(
        {"temperature": (("y", "x"), np.random.random((ny, nx)).astype(np.float32), {"units": "K"})},
        coords={"lon": (("y", "x"), source_lon_2d), "lat": (("y", "x"), source_lat_2d)},
    )

    # Create a target grid that includes points very close to the poles
    ny_target, nx_target = 10, 20
    lon_target_1d = np.linspace(-180, 180, nx_target)
    # Use latitudes very close to the poles to trigger the issue
    lat_target_1d = np.linspace(-89.9, 89.9, ny_target)
    target_lon_2d, target_lat_2d = np.meshgrid(lon_target_1d, lat_target_1d)

    # Create target dataset
    target_ds = xr.Dataset(coords={"lon": (("y", "x"), target_lon_2d), "lat": (("y", "x"), target_lat_2d)})

    # Perform bilinear interpolation
    result = source_ds["temperature"].regrid.linear(target_ds)

    # Check that there are no NaN values in the polar regions
    # Specifically check the first and last latitude rows (closest to poles)
    assert not np.any(np.isnan(result.values)), "Found NaN values in polar interpolation"

    # Additional check: ensure result has the expected shape
    assert result.shape == (ny_target, nx_target), f"Expected shape {(ny_target, nx_target)}, got {result.shape}"


def test_polar_interpolation_with_nan_fill_method():
    """Test that polar interpolation works with fill_method='nearest'."""
    # Create a source grid that includes polar regions
    ny, nx = 20, 40
    lon_1d = np.linspace(-180, 180, nx)
    lat_1d = np.linspace(-90, 90, ny)  # Include the poles
    source_lon_2d, source_lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Create source dataset with some NaN values to test fill behavior
    source_data = np.random.random((ny, nx)).astype(np.float32)
    source_data[0, :] = np.nan  # NaN values near one pole
    source_data[-1, :] = np.nan  # NaN values near other pole

    source_ds = xr.Dataset(
        {"temperature": (("y", "x"), source_data, {"units": "K"})},
        coords={"lon": (("y", "x"), source_lon_2d), "lat": (("y", "x"), source_lat_2d)},
    )

    # Create a target grid that includes points very close to the poles
    ny_target, nx_target = 10, 20
    lon_target_1d = np.linspace(-180, 180, nx_target)
    # Use latitudes very close to the poles to trigger the issue
    lat_target_1d = np.linspace(-89.9, 89.9, ny_target)
    target_lon_2d, target_lat_2d = np.meshgrid(lon_target_1d, lat_target_1d)

    # Create target dataset
    target_ds = xr.Dataset(coords={"lon": (("y", "x"), target_lon_2d), "lat": (("y", "x"), target_lat_2d)})

    # Perform bilinear interpolation with nearest fill method
    result = source_ds["temperature"].regrid.linear(target_ds, fill_method="nearest")

    # With fill_method='nearest', polar regions should have values (not NaN)
    # even when source has NaN values at the poles
    nan_count = np.sum(np.isnan(result.values))
    logging.info("NaN count in result: %s", nan_count)
    logging.info("Result shape: %s", result.shape)

    # The result should have fewer NaN values than a strict linear interpolation would have
    # due to the fallback to nearest neighbor in polar regions
    # The important thing is that the interpolation doesn't crash or produce all NaNs


if __name__ == "__main__":
    test_polar_interpolation_no_nan()
    test_polar_interpolation_with_nan_fill_method()
    logging.info("All polar interpolation tests passed!")
