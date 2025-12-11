"""Tests for the CurvilinearInterpolator class."""

import logging

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from monet_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


def test_curvilinear_interpolator_initialization():
    """Test that CurvilinearInterpolator can be initialized with basic parameters."""
    # Create simple source and target grids
    source_lat = np.linspace(-10, 10, 5)
    source_lon = np.linspace(-10, 10, 6)
    source_lat_2d, source_lon_2d = np.meshgrid(source_lat, source_lon)

    target_lat = np.linspace(-5, 5, 3)
    target_lon = np.linspace(-5, 5, 4)
    target_lat_2d, target_lon_2d = np.meshgrid(target_lat, target_lon)

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat_2d), "longitude": (["y", "x"], source_lon_2d)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat_2d), "longitude": (["y_target", "x_target"], target_lon_2d)}
    )

    # Test initialization with different methods
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    assert interpolator.method == "nearest"

    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="linear")
    assert interpolator.method == "linear"

    # Test with different options
    interpolator = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", spherical=False, fill_method="nearest", extrapolate=True
    )
    assert interpolator.spherical is False
    assert interpolator.fill_method == "nearest"
    assert interpolator.extrapolate is True


def test_curvilinear_interpolator_coordinates_validation():
    """Test that coordinate validation works correctly."""
    # Create grids with proper 2D coordinates
    source_lat = np.linspace(-10, 10, 5)
    source_lon = np.linspace(-10, 10, 6)
    source_lat_2d, source_lon_2d = np.meshgrid(source_lat, source_lon)

    target_lat = np.linspace(-5, 5, 3)
    target_lon = np.linspace(-5, 5, 4)
    target_lat_2d, target_lon_2d = np.meshgrid(target_lat, target_lon)

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat_2d), "longitude": (["y", "x"], source_lon_2d)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat_2d), "longitude": (["y_target", "x_target"], target_lon_2d)}
    )

    # This should work fine
    CurvilinearInterpolator(source_grid, target_grid)

    # Test with 1D coordinates (should now work with our updates)
    rectilinear_source_grid = xr.Dataset({"latitude": (["y"], source_lat), "longitude": (["x"], source_lon)})

    # This should now work with 1D coordinates (rectilinear-to-curvilinear)
    interpolator_1d = CurvilinearInterpolator(rectilinear_source_grid, target_grid)
    assert interpolator_1d.method == "linear"  # Default method

    # Test with mismatched dimensions (should still fail)
    bad_source_grid = xr.Dataset(
        {
            "latitude": (["y"], source_lat),
            "longitude": (["z", "w"], source_lon_2d),  # Different dimension names to avoid conflicts
        }
    )

    with pytest.raises(
        ValueError, match="Source latitude and longitude coordinates must have same number of dimensions"
    ):
        CurvilinearInterpolator(bad_source_grid, target_grid)


def test_curvilinear_interpolator_nearest_interpolation():
    """Test nearest neighbor interpolation."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

    target_x, target_y = np.meshgrid(np.linspace(0, 4, 3), np.linspace(0, 5, 4))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.random.rand(6, 5)  # (y, x)
    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(6), "x": range(5)})

    # Test nearest neighbor interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result = interpolator(test_data)

    # Check result dimensions
    assert result.shape == target_lat.shape
    assert "y_target" in result.dims
    assert "x_target" in result.dims


def test_curvilinear_interpolator_nearest_interpolation_with_time():
    """Test nearest neighbor interpolation with additional dimensions."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0, 2, 2), np.linspace(0, 3, 3))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with time dimension
    time_dim = 5
    data_values = np.random.rand(time_dim, 4, 3)  # (time, y, x)
    test_data = xr.DataArray(
        data_values, dims=["time", "y", "x"], coords={"time": range(time_dim), "y": range(4), "x": range(3)}
    )

    # Test nearest neighbor interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result = interpolator(test_data)

    # Check result dimensions - should have time and target grid dimensions
    expected_shape = (time_dim, 3, 2)  # (time, y_target, x_target)
    assert result.shape == expected_shape
    assert "time" in result.dims
    assert "y_target" in result.dims
    assert "x_target" in result.dims


def test_curvilinear_interpolator_dataset_interpolation():
    """Test interpolation of entire datasets."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(3))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0, 2, 2), np.linspace(0, 2, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test dataset
    data_values = np.random.rand(3, 3)
    test_dataset = xr.Dataset(
        {
            "var1": (["y", "x"], data_values),
            "var2": (["y", "x"], data_values * 2),
            "other_var": (("time",), np.arange(1)),  # This should be preserved as-is
        }
    )

    # Test dataset interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result = interpolator(test_dataset)

    # Check that interpolated variables have correct shape
    assert result["var1"].shape == (2, 2)
    assert result["var2"].shape == (2, 2)
    # Check that non-spatial variable is preserved
    assert "other_var" in result
    np.testing.assert_array_equal(result["other_var"].values, test_dataset["other_var"].values)

    # Check that target coordinates are added
    assert "y_target" in result.coords
    assert "x_target" in result.coords


def test_curvilinear_interpolator_linear_interpolation():
    """Test linear interpolation (basic functionality)."""
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y + 0.0001 * np.random.rand(*source_x.shape)
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.ones((4, 4)) * 5.0  # Simple constant data
    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(4), "x": range(4)})

    # Test linear interpolation
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="linear")
    result = interpolator(test_data)

    # With constant data, result should be approximately the same value
    assert result.shape == target_lat.shape
    # Values should be close to 5.0 (the original constant value)
    np.testing.assert_allclose(result.data, 5.0, rtol=1e-5)


if __name__ == "__main__":
    test_curvilinear_interpolator_initialization()
    test_curvilinear_interpolator_coordinates_validation()
    test_curvilinear_interpolator_nearest_interpolation()
    test_curvilinear_interpolator_nearest_interpolation_with_time()
    test_curvilinear_interpolator_dataset_interpolation()
    test_curvilinear_interpolator_linear_interpolation()
    logging.info("All tests passed!")
