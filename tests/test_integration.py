"""Integration tests for the complete monet-regrid rebranding pipeline."""

import logging

import numpy as np
import pytest
import xarray as xr

from monet_regrid.constants import GridType
from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder
from monet_regrid.utils import _get_grid_type

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old imports: from xarray_regrid.core import ...; from xarray_regrid.constants import ...; from xarray_regrid.utils import ...
# New imports: from monet_regrid.core import ...; from monet_regrid.constants import ...; from monet_regrid.utils import ...


def test_rectilinear_to_rectilinear_regridding():
    """Test that rectilinear-to-rectilinear regridding still works as before."""
    # Create source data on a rectilinear grid
    source_lat = np.linspace(-10, 10, 20)
    source_lon = np.linspace(-20, 20, 30)
    source_data = xr.DataArray(
        np.random.random((20, 30)), dims=["lat", "lon"], coords={"lat": source_lat, "lon": source_lon}
    )

    # Create target grid
    target_lat = np.linspace(-8, 8, 15)
    target_lon = np.linspace(-15, 15, 25)
    target_grid = xr.Dataset({"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)})

    # Use the regrid accessor (this should use RectilinearRegridder)
    result = source_data.regrid.linear(target_grid)

    # Check that result has expected dimensions
    assert result.shape == (15, 25)  # target grid shape
    assert "lat" in result.coords
    assert "lon" in result.coords
    np.testing.assert_array_equal(result.coords["lat"], target_lat)
    np.testing.assert_array_equal(result.coords["lon"], target_lon)


def test_curvilinear_to_curvilinear_regridding():
    """Test curvilinear-to-curvilinear regridding with CurvilinearRegridder."""
    # Create curvilinear source grid (2D lat/lon coordinates)
    source_x, source_y = np.meshgrid(np.arange(10), np.arange(12))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
    source_lon = -10 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

    # Create curvilinear target grid
    target_x, target_y = np.meshgrid(np.linspace(0, 9, 7), np.linspace(0, 11, 8))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    # Create source data
    source_data = xr.DataArray(
        np.random.random((12, 10)),  # (y, x)
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Create target grid dataset
    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Check grid types
    source_type = _get_grid_type(
        xr.Dataset({"latitude": source_data["latitude"], "longitude": source_data["longitude"]})
    )
    target_type = _get_grid_type(target_grid)

    assert source_type == GridType.CURVILINEAR
    assert target_type == GridType.CURVILINEAR

    # Use the regrid accessor (this should use CurvilinearRegridder)
    result = source_data.regrid.linear(target_grid)

    # Check that result has expected dimensions
    assert result.shape == (8, 7)  # target grid shape (y_target, x_target)
    assert "latitude" in result.coords
    assert "longitude" in result.coords


def test_rectilinear_to_curvilinear_regridding():
    """Test rectilinear-to-curvilinear regridding."""
    # Create rectilinear source data
    source_lat = np.linspace(-10, 10, 20)
    source_lon = np.linspace(-20, 20, 30)
    source_data = xr.DataArray(
        np.random.random((20, 30)), dims=["lat", "lon"], coords={"lat": source_lat, "lon": source_lon}
    )

    # Create curvilinear target grid
    target_x, target_y = np.meshgrid(np.linspace(0, 19, 15), np.linspace(0, 29, 20))
    target_lat = -10 + 1.0 * target_x + 0.05 * target_y
    target_lon = -20 + 2.0 * target_x + 0.1 * target_y

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Check target grid type
    target_type = _get_grid_type(target_grid)
    assert target_type == GridType.CURVILINEAR

    # Use the regrid accessor (this should use CurvilinearRegridder)
    result = source_data.regrid.linear(target_grid)

    # Check that result has expected dimensions
    assert result.shape == (20, 15)  # target grid shape (y_target, x_target)
    assert "latitude" in result.coords
    assert "longitude" in result.coords


def test_curvilinear_to_rectilinear_regridding():
    """Test curvilinear-to-rectilinear regridding."""
    # Create curvilinear source grid
    source_x, source_y = np.meshgrid(np.arange(10), np.arange(12))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    source_data = xr.DataArray(
        np.random.random((12, 10)),  # (y, x)
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Create rectilinear target grid
    target_lat = np.linspace(30, 36, 15)
    target_lon = np.linspace(-100, -95, 20)
    target_grid = xr.Dataset({"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)})

    # Check source grid type
    source_type = _get_grid_type(
        xr.Dataset({"latitude": source_data["latitude"], "longitude": source_data["longitude"]})
    )
    assert source_type == GridType.CURVILINEAR

    # Use the regrid accessor (this should use CurvilinearRegridder)
    result = source_data.regrid.linear(target_grid)

    # Check that result has expected dimensions
    assert result.shape == (15, 20)  # target grid shape
    assert "lat" in result.coords
    assert "lon" in result.coords
    np.testing.assert_array_equal(result.coords["lat"], target_lat)
    np.testing.assert_array_equal(result.coords["lon"], target_lon)


def test_backward_compatibility():
    """Test that old API still works identically."""
    # Create source data on a rectilinear grid
    source_lat = np.linspace(-10, 10, 20)
    source_lon = np.linspace(-20, 20, 30)
    source_data = xr.DataArray(
        np.random.random((20, 30)), dims=["lat", "lon"], coords={"lat": source_lat, "lon": source_lon}
    )

    # Create target grid
    target_lat = np.linspace(-8, 8, 15)
    target_lon = np.linspace(-15, 15, 25)
    target_grid = xr.Dataset({"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)})

    # Test that the old API still works
    result1 = source_data.regrid.linear(target_grid)

    # Also test with build_regridder method
    regridder = source_data.regrid.build_regridder(target_grid, method="linear")
    result2 = regridder()

    # Results should be identical
    np.testing.assert_allclose(result1.values, result2.values, rtol=1e-10)


def test_different_methods_curvilinear():
    """Test different interpolation methods with curvilinear grids."""
    # Create curvilinear source grid
    source_x, source_y = np.meshgrid(np.arange(8), np.arange(10))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    source_data = xr.DataArray(
        np.random.random((10, 8)),  # (y, x)
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Create curvilinear target grid
    target_x, target_y = np.meshgrid(np.linspace(0, 7, 5), np.linspace(0, 9, 6))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Test different methods
    result_nearest = source_data.regrid.nearest(target_grid)
    result_linear = source_data.regrid.linear(target_grid)

    # Both should have the same shape
    assert result_nearest.shape == (6, 5)
    assert result_linear.shape == (6, 5)

    # Both should have the correct coordinates
    assert "latitude" in result_nearest.coords
    assert "longitude" in result_nearest.coords
    assert "latitude" in result_linear.coords
    assert "longitude" in result_linear.coords


def test_grid_detection_accuracy():
    """Test that grid type detection works correctly."""
    # Test rectilinear detection
    rect_data = xr.DataArray(
        np.random.random((10, 20)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-10, 10, 10), "lon": np.linspace(-20, 20, 20)},
    )

    rect_grid = xr.Dataset({"lat": (["lat"], np.linspace(-5, 5, 8)), "lon": (["lon"], np.linspace(-10, 10, 12))})

    rect_source_type = _get_grid_type(xr.Dataset({"lat": rect_data["lat"], "lon": rect_data["lon"]}))
    rect_target_type = _get_grid_type(rect_grid)

    assert rect_source_type == GridType.RECTILINEAR
    assert rect_target_type == GridType.RECTILINEAR

    # Test curvilinear detection
    source_x, source_y = np.meshgrid(np.arange(8), np.arange(10))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    curv_data = xr.DataArray(
        np.random.random((10, 8)),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    target_x, target_y = np.meshgrid(np.linspace(0, 7, 5), np.linspace(0, 9, 6))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    curv_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    curv_source_type = _get_grid_type(
        xr.Dataset({"latitude": curv_data["latitude"], "longitude": curv_data["longitude"]})
    )
    curv_target_type = _get_grid_type(curv_grid)

    assert curv_source_type == GridType.CURVILINEAR
    assert curv_target_type == GridType.CURVILINEAR


if __name__ == "__main__":
    test_rectilinear_to_rectilinear_regridding()
    test_curvilinear_to_curvilinear_regridding()
    test_rectilinear_to_curvilinear_regridding()
    test_curvilinear_to_rectilinear_regridding()
    test_backward_compatibility()
    test_different_methods_curvilinear()
    test_grid_detection_accuracy()
    logging.info("All integration tests passed!")
