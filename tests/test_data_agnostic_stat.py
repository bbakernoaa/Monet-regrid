import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def test_rectilinear_data_agnostic_stat():
    """Test statistical regridding with RectilinearRegridder when data is passed at call-time."""
    # Define source grid
    source_lat = np.linspace(-90, 90, 100)
    source_lon = np.linspace(-180, 180, 200)
    source_da = xr.DataArray(
        np.ones((100, 200)), coords={"lat": source_lat, "lon": source_lon}, dims=["lat", "lon"], name="test_data"
    )

    # Define target grid (coarser)
    target_lat = np.linspace(-90, 90, 10)
    target_lon = np.linspace(-180, 180, 20)
    target_ds = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # Initialize regridder without data
    regridder = RectilinearRegridder(source_data=None, target_grid=target_ds)

    # Regrid passing data at call-time
    result = regridder.stat(method="mean", data=source_da)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (10, 20)
    assert np.allclose(result.values, 1.0)
    assert "Reduced using monet_regrid.methods.flox_reduce.statistic_reduce" in result.attrs["history"]


def test_curvilinear_data_agnostic_stat():
    """Test statistical regridding with CurvilinearRegridder when data is passed at call-time."""
    # Define curvilinear source grid
    y, x = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    lat = xr.DataArray(y, dims=["y", "x"])
    lon = xr.DataArray(x, dims=["y", "x"])

    source_da = xr.DataArray(np.ones((50, 50)), coords={"lat": lat, "lon": lon}, dims=["y", "x"], name="test_data")

    # Define target grid (rectilinear)
    target_lat = np.linspace(0, 10, 5)
    target_lon = np.linspace(0, 10, 5)
    target_ds = xr.Dataset(coords={"latitude": target_lat, "longitude": target_lon})

    # Initialize regridder without data
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_ds)

    # Regrid passing data at call-time
    result = regridder.stat(method="mean", data=source_da)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (5, 5)
    assert np.allclose(result.values, 1.0)
    assert "Reduced using monet_regrid.methods.flox_reduce.statistic_reduce" in result.attrs["history"]


def test_rectilinear_data_agnostic_mode():
    """Test mode regridding with RectilinearRegridder when data is passed at call-time."""
    source_lat = np.linspace(0, 10, 20)
    source_lon = np.linspace(0, 10, 20)
    # Categorical data: 1 everywhere
    source_da = xr.DataArray(
        np.ones((20, 20), dtype=int), coords={"lat": source_lat, "lon": source_lon}, dims=["lat", "lon"], name="test_data"
    )

    target_lat = np.linspace(0, 10, 5)
    target_lon = np.linspace(0, 10, 5)
    target_ds = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    regridder = RectilinearRegridder(source_data=None, target_grid=target_ds)

    values = np.array([0, 1, 2])
    result = regridder.most_common(values=values, data=source_da)

    assert result.dtype == int
    assert result.shape == (5, 5)
    assert np.all(result.values == 1)
