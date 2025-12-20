import numpy as np
import pytest
import xarray as xr

from monet_regrid import CurvilinearRegridder, RectilinearRegridder
from monet_regrid.core import BaseRegridder

"""Tests for the new regridder class hierarchy."""


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


def test_rectilinear_regridder_to_netcdf(tmp_path):
    """Test saving and loading RectilinearRegridder to/from NetCDF."""
    source_data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(20)},
    )
    target_grid = xr.Dataset(
        coords={"y": np.linspace(0, 9, 5), "x": np.linspace(0, 19, 10)}
    )
    regridder = RectilinearRegridder(
        source_data, target_grid, method="linear", time_dim=None
    )

    filepath = tmp_path / "test_regridder.nc"
    regridder.to_netcdf(filepath)
    loaded_regridder = RectilinearRegridder.from_netcdf(filepath)

    xr.testing.assert_allclose(regridder.source_data, loaded_regridder.source_data)
    xr.testing.assert_allclose(regridder.target_grid, loaded_regridder.target_grid)
    assert regridder.method == loaded_regridder.method
    assert regridder.time_dim == loaded_regridder.time_dim

    # Test that the loaded regridder produces the same result
    expected_result = regridder()
    loaded_result = loaded_regridder()
    xr.testing.assert_allclose(expected_result, loaded_result)


def test_curvilinear_regridder_to_netcdf(tmp_path):
    """Test saving and loading CurvilinearRegridder to/from NetCDF."""
    source_x, source_y = np.meshgrid(np.arange(10), np.arange(10))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y
    source_data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["y", "x"],
        coords={
            "latitude": (("y", "x"), source_lat),
            "longitude": (("y", "x"), source_lon),
        },
    )

    target_x, target_y = np.meshgrid(np.arange(8), np.arange(8))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y
    target_grid = xr.Dataset(
        coords={
            "latitude": (("y_out", "x_out"), target_lat),
            "longitude": (("y_out", "x_out"), target_lon),
        }
    )

    regridder = CurvilinearRegridder(source_data, target_grid, method="linear")

    filepath = tmp_path / "test_regridder.nc"
    regridder.to_netcdf(filepath)
    loaded_regridder = CurvilinearRegridder.from_netcdf(filepath)

    xr.testing.assert_allclose(regridder.source_data, loaded_regridder.source_data)
    xr.testing.assert_allclose(regridder.target_grid, loaded_regridder.target_grid)
    assert regridder.method == loaded_regridder.method

    # Test that the loaded regridder produces the same result
    expected_result = regridder()
    loaded_result = loaded_regridder()
    xr.testing.assert_allclose(expected_result, loaded_result)


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
