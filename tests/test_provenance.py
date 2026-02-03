import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import RectilinearRegridder
from monet_regrid.curvilinear import CurvilinearInterpolator


@pytest.fixture
def source_grid():
    lon = np.linspace(0, 10, 5)
    lat = np.linspace(0, 10, 5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        }
    )
    return ds


@pytest.fixture
def target_grid():
    lon = np.linspace(0, 10, 3)
    lat = np.linspace(0, 10, 3)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        }
    )
    return ds


def test_curvilinear_provenance_dataarray(source_grid, target_grid):
    data = xr.DataArray(
        np.random.rand(5, 5),
        coords=source_grid.coords,
        dims=("y", "x"),
        name="test_data",
    )
    data.attrs["history"] = "Original data"

    interpolator = CurvilinearInterpolator(source_grid, target_grid, "lat", "lon", "lat", "lon", method="nearest")
    result = interpolator(data)

    assert "history" in result.attrs
    assert "Original data" in result.attrs["history"]
    assert "Interpolated using monet_regrid.curvilinear.CurvilinearInterpolator (method=nearest)" in result.attrs["history"]


def test_curvilinear_provenance_dataset(source_grid, target_grid):
    ds = xr.Dataset(
        {"test_data": (("y", "x"), np.random.rand(5, 5))},
        coords=source_grid.coords,
    )
    ds.attrs["history"] = "Original dataset"

    interpolator = CurvilinearInterpolator(source_grid, target_grid, "lat", "lon", "lat", "lon", method="nearest")
    result = interpolator(ds)

    assert "history" in result.attrs
    assert "Original dataset" in result.attrs["history"]
    assert "Interpolated using monet_regrid.curvilinear.CurvilinearInterpolator (method=nearest)" in result.attrs["history"]


def test_rectilinear_provenance(source_grid, target_grid):
    # Convert grids to rectilinear for this test
    lon_1d = source_grid.lon.values[0, :]
    lat_1d = source_grid.lat.values[:, 0]
    source_rect = xr.Dataset(coords={"lat": lat_1d, "lon": lon_1d})

    lon_t_1d = target_grid.lon.values[0, :]
    lat_t_1d = target_grid.lat.values[:, 0]
    target_rect = xr.Dataset(coords={"lat": lat_t_1d, "lon": lon_t_1d})

    data = xr.DataArray(
        np.random.rand(len(lat_1d), len(lon_1d)),
        coords=source_rect.coords,
        dims=("lat", "lon"),
        name="test_data",
    )
    data.attrs["history"] = "Original data"

    regridder = RectilinearRegridder(data, target_rect, method="nearest")
    result = regridder()

    assert "history" in result.attrs
    assert "Original data" in result.attrs["history"]
