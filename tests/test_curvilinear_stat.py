"""
Unit tests for curvilinear statistical regridding.

This file is part of monet-regrid.
"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import monet_regrid  # noqa: F401


@pytest.fixture
def curvilinear_ds():
    """Create a sample curvilinear dataset."""
    ny, nx = 10, 10
    lon1d = np.linspace(0, 10, nx)
    lat1d = np.linspace(0, 10, ny)
    lon, lat = np.meshgrid(lon1d, lat1d)
    data = np.ones((ny, nx))

    ds = xr.Dataset(
        {"data": (("y", "x"), data)},
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        },
    )
    return ds


@pytest.fixture
def target_rectilinear_grid():
    """Create a target rectilinear grid."""
    return xr.Dataset(
        coords={
            "lat": (("lat",), np.array([2.0, 5.0, 8.0]), {"units": "degrees_north"}),
            "lon": (("lon",), np.array([2.0, 5.0, 8.0]), {"units": "degrees_east"}),
        }
    )


def test_curvilinear_stat_mean(curvilinear_ds, target_rectilinear_grid):
    """Test that curvilinear stat mean works correctly."""
    result = curvilinear_ds.regrid.stat(target_rectilinear_grid, method="mean")

    assert result["data"].shape == (3, 3)
    assert np.allclose(result["data"].values, 1.0)
    assert "lat" in result.dims
    assert "lon" in result.dims
    assert result["lat"].attrs["units"] == "degrees_north"


def test_curvilinear_most_common(curvilinear_ds, target_rectilinear_grid):
    """Test that curvilinear most_common works correctly."""
    # Modify data to have categories
    data = curvilinear_ds["data"].values.copy().astype(int)
    data[:5, :5] = 1
    data[5:, 5:] = 2
    curvilinear_ds["data"] = (("y", "x"), data)

    values = np.array([0, 1, 2])
    result = curvilinear_ds["data"].regrid.most_common(target_rectilinear_grid, values=values)

    assert result.shape == (3, 3)
    assert result.sel(lat=2.0, lon=2.0) == 1
    assert result.sel(lat=8.0, lon=8.0) == 2


def test_curvilinear_stat_lazy(target_rectilinear_grid):
    """Test that curvilinear stat is lazy when using Dask."""
    ny, nx = 10, 10
    lon1d = np.linspace(0, 10, nx)
    lat1d = np.linspace(0, 10, ny)
    lon, lat = np.meshgrid(lon1d, lat1d)

    lazy_data = da.ones((ny, nx), chunks=(5, 5))
    ds = xr.Dataset(
        {"data": (("y", "x"), lazy_data)},
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        },
    )

    result = ds.regrid.stat(target_rectilinear_grid, method="mean")

    assert isinstance(result["data"].data, da.Array)

    # Compute and verify
    computed = result.compute()
    assert np.allclose(computed["data"].values, 1.0)


def test_curvilinear_stat_provenance(curvilinear_ds, target_rectilinear_grid):
    """Test that history is updated for curvilinear stat."""
    result = curvilinear_ds.regrid.stat(target_rectilinear_grid, method="sum")
    assert "Reduced using monet_regrid.methods.flox_reduce.statistic_reduce" in result.attrs["history"]
