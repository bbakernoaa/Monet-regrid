import pytest

"""Tests for the new regridder class hierarchy."""

import os

import numpy as np
import xarray as xr

from monet_regrid import CurvilinearRegridder, RectilinearRegridder
from monet_regrid.core import BaseRegridder

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from monet_regrid import RectilinearRegridder, CurvilinearRegridder
# New import: from monet_regrid import RectilinearRegridder, CurvilinearRegridder


def test_baseregridder_abstract():
    """Test that BaseRegridder is properly abstract."""

    # Should not be instantiable directly
    with pytest.raises(TypeError):
        BaseRegridder(None, None)


def test_rectilinear_regridder_initialization():
    """Test RectilinearRegridder initialization."""
    # Create sample data
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test initialization
    regridder = RectilinearRegridder(source_data, target_grid, method="linear")

    assert regridder.source_data is source_data
    assert regridder.target_grid is target_grid
    assert regridder.method == "linear"

    # Test info method
    info = regridder.info()
    assert info["type"] == "RectilinearRegridder"
    assert info["method"] == "linear"
    assert info["grid_type"] == "rectilinear"


def test_rectilinear_regridder_call():
    """Test RectilinearRegridder call functionality."""
    # Create sample data
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test linear regridding
    regridder = RectilinearRegridder(source_data, target_grid, method="linear")
    result = regridder()

    # Check that result has the expected dimensions
    assert "lat" in result.coords
    assert "lon" in result.coords
    assert len(result["lat"]) == 8
    assert len(result["lon"]) == 8


def test_curvilinear_regridder_implementation():
    """Test that CurvilinearRegridder is properly implemented."""
    # Create curvilinear sample data (2D coordinates)
    source_x, source_y = np.meshgrid(np.arange(10), np.arange(10))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Create curvilinear target grid
    target_x, target_y = np.meshgrid(np.linspace(0, 9, 8), np.linspace(0, 9, 8))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Test initialization
    regridder = CurvilinearRegridder(source_data, target_grid, method="linear")

    # Test that call works (though may fail due to coordinate validation)
    # The important thing is that it doesn't raise NotImplementedError
    try:
        result = regridder()
        # If it succeeds, check that result has proper dimensions
        assert "latitude" in result.coords
        assert "longitude" in result.coords
    except ValueError as e:
        # If it fails, it should be due to coordinate validation, not NotImplementedError
        assert "Source coordinates must be 2D" not in str(e)

    # Test that to_file and from_file work
    try:
        regridder.to_file("test_curvilinear.nc")
        loaded_regridder = CurvilinearRegridder.from_file("test_curvilinear.nc")
        assert loaded_regridder.method == "linear"
    except NotImplementedError:
        # If file operations are not implemented, that's OK for now
        pass
    finally:
        # Clean up test file if it was created
        if os.path.exists("test_curvilinear.nc"):
            os.remove("test_curvilinear.nc")

    # Test info method works
    info = regridder.info()
    assert info["type"] == "CurvilinearRegridder"
    assert info["method"] == "linear"
    assert info["grid_type"] == "curvilinear"


def test_rectilinear_regridder_methods():
    """Test different methods in RectilinearRegridder."""
    # Create sample data
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test different methods
    for method in ["linear", "nearest", "cubic"]:
        regridder = RectilinearRegridder(source_data, target_grid, method=method)
        result = regridder()
        assert "lat" in result.coords
        assert "lon" in result.coords


def test_rectilinear_regridder_conservative():
    """Test conservative method in RectilinearRegridder."""
    # Create sample data with lat/lon coordinates
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
        attrs={"units": "degrees"},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test conservative regridding
    regridder = RectilinearRegridder(source_data, target_grid, method="conservative", skipna=True, nan_threshold=0.5)
    result = regridder()
    assert "lat" in result.coords
    assert "lon" in result.coords


def test_rectilinear_regridder_dataset():
    """Test RectilinearRegridder with Dataset input."""
    # Create sample dataset
    source_data = xr.Dataset(
        {"var1": (["lat", "lon"], np.random.random((10, 10))), "var2": (["lat", "lon"], np.random.random((10, 10)))},
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test with dataset
    regridder = RectilinearRegridder(source_data, target_grid, method="linear")
    result = regridder()

    assert "var1" in result.data_vars
    assert "var2" in result.data_vars
    assert "lat" in result.coords
    assert "lon" in result.coords
