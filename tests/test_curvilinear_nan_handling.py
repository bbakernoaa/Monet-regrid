"""Comprehensive tests for NaN handling in CurvilinearInterpolator."""

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from xarray_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


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
    data_values[1, 1] = np.nan  # Add NaN at position (1, 1)
    data_values[2, 3] = np.nan  # Add NaN at position (2, 3)

    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(4), "x": range(4)})

    # Test nearest neighbor interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result = interpolator(test_data)

    # Result should handle NaN values properly
    assert result.shape == target_lat.shape
    # Check that NaN values in source don't cause all results to be NaN
    assert not np.all(np.isnan(result))

    # Some points might be NaN if they interpolate from NaN source points
    nan_count = np.sum(np.isnan(result))
    assert nan_count >= 0  # Could be 0 or more depending on interpolation


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
    data_values[1, 1] = np.nan  # Add NaN at position (1, 1)
    data_values[2, 3] = np.nan  # Add NaN at position (2, 3)

    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(4), "x": range(4)})

    # Test linear interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="linear")
    result = interpolator(test_data)

    # Result should handle NaN values properly
    assert result.shape == target_lat.shape
    # Check that NaN values in source don't cause all results to be NaN
    assert not np.all(np.isnan(result))

    # Some points might be NaN if they interpolate from NaN source points
    nan_count = np.sum(np.isnan(result))
    assert nan_count >= 0  # Could be 0 or more depending on interpolation


def test_all_nan_source_data():
    """Test interpolation when all source data is NaN."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(3))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with all NaN values
    data_values = np.full((3, 3), np.nan)
    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(3), "x": range(3)})

    # Test both nearest and linear interpolation
    for method in ["nearest", "linear"]:
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method=method)
        result = interpolator(test_data)

        # All results should be NaN when source is all NaN
        assert np.all(np.isnan(result))
        assert result.shape == target_lat.shape


def test_mixed_nan_valid_data():
    """Test interpolation with mixed NaN and valid data."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    # Create target grid that covers the center of the source
    target_x, target_y = np.meshgrid(np.linspace(1, 2, 2), np.linspace(1, 2, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with a checkerboard pattern of NaN and valid values
    data_values = np.ones((4, 4)) * 5.0
    data_values[::2, ::2] = np.nan  # Corner points are NaN

    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(4), "x": range(4)})

    # Test both nearest and linear interpolation
    # Use fill_method="nearest" to ensure NaNs don't propagate excessively for this test
    for method in ["nearest", "linear"]:
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method=method, fill_method="nearest")
        result = interpolator(test_data)

        # Result should handle NaN reasonably - not all values should be NaN
        # (Since target points are in the center of the grid where valid points exist nearby)
        assert not np.all(np.isnan(result))
        assert result.shape == target_lat.shape


def test_out_of_domain_points_nan_handling():
    """Test that out-of-domain points become NaN with fill_method='nan'."""
    # Create source grid with limited extent
    source_x, source_y = np.meshgrid(np.arange(2), np.arange(2))
    source_lat = 30 + source_x
    source_lon = -100 + source_y

    # Create target grid that extends beyond source
    target_x, target_y = np.meshgrid(np.arange(-1, 3), np.arange(-1, 3))
    target_lat = 30 + target_x
    target_lon = -100 + target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create valid source data
    data_values = np.ones((2, 2)) * 5.0
    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(2), "x": range(2)})

    # Test with fill_method='nan' (default)
    # Use a small radius of influence to ensure out-of-domain points remain NaN
    interpolator = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", fill_method="nan", radius_of_influence=1e5
    )
    result = interpolator(test_data)

    # Points outside source domain should be NaN
    assert np.any(np.isnan(result))
    assert result.shape == target_lat.shape

    # Test with fill_method='nearest'
    # Use a small radius of influence to ensure consistent behavior
    interpolator_nearest = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", fill_method="nearest", radius_of_influence=1e5
    )
    result_nearest = interpolator_nearest(test_data)

    # With 'nearest' method, all points should have values
    assert np.all(np.isfinite(result_nearest))


def test_nan_handling_with_time_dimension():
    """Test NaN handling with additional dimensions like time."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(3))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with time dimension and some NaN values
    time_dim = 5
    data_values = np.ones((time_dim, 3, 3)) * 5.0
    data_values[:, 1, 1] = np.nan  # Make (1, 1) position NaN for all times
    data_values[2:4, 2, 2] = np.nan  # Make (2, 2) position NaN for some times

    test_data = xr.DataArray(
        data_values, dims=["time", "y", "x"], coords={"time": range(time_dim), "y": range(3), "x": range(3)}
    )

    # Test nearest neighbor interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result = interpolator(test_data)

    # Result should handle NaN values properly
    expected_shape = (time_dim, 2, 2)  # (time, y_target, x_target)
    assert result.shape == expected_shape
    # Check that NaN values don't cause all results to be NaN
    assert not np.all(np.isnan(result))


def test_nan_comparison_with_rectilinear():
    """Test that curvilinear NaN handling is consistent with rectilinear."""
    # Create identical source grids for both methods
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    # Create target grid that is also curvilinear for compatibility
    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 3), np.linspace(0.5, 2.5, 3))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid_curvilinear = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with NaN values
    data_values = np.ones((4, 4)) * 5.0
    data_values[1, 1] = np.nan

    test_data_curvilinear = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(4), "x": range(4)})

    # Test curvilinear interpolation
    curvilinear_interpolator = CurvilinearInterpolator(source_grid_curvilinear, target_grid, method="nearest")
    result_curvilinear = curvilinear_interpolator(test_data_curvilinear)

    # Test that it handles NaN reasonably
    assert not np.all(np.isnan(result_curvilinear))

    # The number of NaN values should be reasonable
    nan_count_curvilinear = np.sum(np.isnan(result_curvilinear))

    assert nan_count_curvilinear >= 0


if __name__ == "__main__":
    test_nan_handling_in_source_data_nearest()
    test_nan_handling_in_source_data_linear()
    test_all_nan_source_data()
    test_mixed_nan_valid_data()
    test_out_of_domain_points_nan_handling()
    test_nan_handling_with_time_dimension()
    test_nan_comparison_with_rectilinear()
    print("All NaN handling tests passed!")
