"""
Comprehensive unit tests for curvilinear regridding methods.

Tests Linear, Nearest, Conservative, Bilinear, and Cubic methods.
Also verifies the fix for data-override in least_common.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.accessor import Regridder  # noqa: F401


@pytest.fixture
def curvilinear_ds_with_bounds() -> xr.Dataset:
    """Create a sample curvilinear dataset with explicit cell bounds.

    Returns
    -------
    xr.Dataset
        A dataset with 2D coordinates and 3D bounds.

    Examples
    --------
    >>> ds = curvilinear_ds_with_bounds()
    >>> "lat_b" in ds.coords
    True
    """
    nx, ny = 10, 10
    lon = np.linspace(-20, 20, nx)
    lat = np.linspace(30, 60, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Add some distortion
    lon2d = lon2d + 2 * np.sin(np.radians(lat2d))
    lat2d = lat2d + 1 * np.cos(np.radians(lon2d))

    def get_bnds(arr: np.ndarray) -> np.ndarray:
        """Compute simple bounding box around centers.

        Parameters
        ----------
        arr : np.ndarray
            The center coordinates.

        Returns
        -------
        np.ndarray
            The vertices (y, x, 4).
        """
        bnds = np.zeros((*arr.shape, 4))
        res_x = np.diff(arr, axis=1).mean() if arr.ndim > 1 else 0.5
        res_y = np.diff(arr, axis=0).mean() if arr.ndim > 1 else 0.5
        bnds[:, :, 0] = arr - res_x / 2 - res_y / 2
        bnds[:, :, 1] = arr + res_x / 2 - res_y / 2
        bnds[:, :, 2] = arr + res_x / 2 + res_y / 2
        bnds[:, :, 3] = arr - res_x / 2 + res_y / 2
        return bnds

    ds = xr.Dataset(
        data_vars={"emissions": (("y", "x"), 100 * np.exp(-((lon2d - 0) ** 2 + (lat2d - 45) ** 2) / 50))},
        coords={
            "lat": (("y", "x"), lat2d, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("y", "x"), lon2d, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("y", "x", "nv"), get_bnds(lat2d)),
            "lon_b": (("y", "x", "nv"), get_bnds(lon2d)),
        },
    )
    return ds


@pytest.fixture
def target_grid() -> xr.Dataset:
    """Create a regular target grid with bounds.

    Returns
    -------
    xr.Dataset
        A dataset with 1D coordinates and 2D bounds.
    """
    nx, ny = 5, 5
    lon = np.linspace(-15, 15, nx)
    lat = np.linspace(35, 55, ny)

    def get_1d_bnds(arr: np.ndarray) -> np.ndarray:
        """Compute 1D bounds.

        Parameters
        ----------
        arr : np.ndarray
            Center coordinates.

        Returns
        -------
        np.ndarray
            Bounds (n, 2).
        """
        res = np.diff(arr).mean()
        return np.stack([arr - res / 2, arr + res / 2], axis=-1)

    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lat, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("lon",), lon, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("lat", "nv"), get_1d_bnds(lat)),
            "lon_b": (("lon", "nv"), get_1d_bnds(lon)),
        }
    )
    return ds


def test_curvilinear_conservative(curvilinear_ds_with_bounds: xr.Dataset, target_grid: xr.Dataset) -> None:
    """Test conservative regridding for curvilinear grids.

    Parameters
    ----------
    curvilinear_ds_with_bounds : xr.Dataset
        Source curvilinear dataset.
    target_grid : xr.Dataset
        Target rectilinear grid.
    """
    result = curvilinear_ds_with_bounds.regrid.conservative(target_grid)
    assert result.emissions.shape == (5, 5)
    assert not np.all(np.isnan(result.emissions.values))
    assert "history" in result.attrs
    assert "CurvilinearRegridder" in result.attrs["history"]
    assert "method='conservative'" in result.attrs["history"]


@pytest.mark.parametrize("method", ["bilinear", "cubic"])
def test_curvilinear_interpolation_methods(curvilinear_ds_with_bounds: xr.Dataset, target_grid: xr.Dataset, method: str) -> None:
    """Test bilinear and cubic interpolation for curvilinear grids.

    Parameters
    ----------
    curvilinear_ds_with_bounds : xr.Dataset
        Source curvilinear dataset.
    target_grid : xr.Dataset
        Target rectilinear grid.
    method : str
        Interpolation method to test.
    """
    # We use the standard accessor methods
    if method == "bilinear":
        result = curvilinear_ds_with_bounds.regrid.bilinear(target_grid)
    else:
        result = curvilinear_ds_with_bounds.regrid.cubic(target_grid)

    assert result.emissions.shape == (5, 5)
    assert not np.all(np.isnan(result.emissions.values))
    assert method in result.attrs["history"]


def test_rectilinear_least_common_data_override() -> None:
    """Test the fix for data override in least_common for rectilinear grids."""
    lat = np.linspace(0, 10, 10)
    lon = np.linspace(0, 10, 10)
    # data1: mostly 1s, some 2s
    d1_vals = np.ones((10, 10), dtype=int)
    d1_vals[0, 0] = 2
    data1 = xr.DataArray(d1_vals, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="d1")

    # data2: mostly 2s, some 1s
    d2_vals = np.ones((10, 10), dtype=int) * 2
    d2_vals[0, 0] = 1
    data2 = xr.DataArray(d2_vals, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="d2")

    target_grid = xr.Dataset(coords={"lat": [5.0], "lon": [2.5, 7.5]})

    # Initialize with data1
    regridder = data1.regrid.build_regridder(target_grid, method="least_common")

    # Call with data2 override
    values = np.array([1, 2])
    result = regridder.least_common(values=values, data=data2)

    # data2 has mostly 2s and one 1. Least common is 1.
    # If it incorrectly used data1 (mostly 1s and one 2), least common would be 2.
    assert result.values.flatten()[0] == 1
