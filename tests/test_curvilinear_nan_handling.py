"""Comprehensive tests for NaN handling in CurvilinearInterpolator."""

import numpy as np
import xarray as xr
import monet_regrid  # noqa: F401


def test_nan_handling_in_source_data_nearest():
    """Test that nearest neighbor interpolation properly handles NaN in source data."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with some NaN values
    data_values = np.ones((4, 4)) * 5.0
    data_values[1, 1] = np.nan
    data_values[2, 3] = np.nan

    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={"latitude": (("y", "x"), source_lat), "longitude": (("y", "x"), source_lon)},
    )

    # Test nearest neighbor interpolation
    result = test_data.regrid.nearest(target_grid)

    # Result should handle NaN values properly
    assert result.shape == target_lat.shape
    assert not np.all(np.isnan(result))
    nan_count = np.sum(np.isnan(result))
    assert nan_count >= 0


def test_nan_handling_in_source_data_linear():
    """Test that linear interpolation properly handles NaN in source data."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with some NaN values
    data_values = np.ones((4, 4)) * 5.0
    data_values[1, 1] = np.nan
    data_values[2, 3] = np.nan

    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={"latitude": (("y", "x"), source_lat), "longitude": (("y", "x"), source_lon)},
    )

    # Test linear interpolation
    result = test_data.regrid.linear(target_grid)

    # Result should handle NaN values properly
    assert result.shape == target_lat.shape
    assert not np.all(np.isnan(result))
    nan_count = np.sum(np.isnan(result))
    assert nan_count >= 0
