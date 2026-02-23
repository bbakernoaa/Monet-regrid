import numpy as np
import pytest
import xarray as xr
import dask.array as da
from monet_regrid.core import RectilinearRegridder, CurvilinearRegridder

def create_rectilinear_data():
    lat = np.arange(-90, 91, 1)
    lon = np.arange(-180, 181, 1)
    data = da.random.random((len(lat), len(lon)), chunks=(91, 181))
    da_source = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="test_data"
    )

    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(-180, 181, 10)
    ds_target = xr.Dataset(
        coords={"lat": target_lat, "lon": target_lon}
    )
    return da_source, ds_target

def create_curvilinear_data():
    # Simple curvilinear grid (can be just 2D lats/lons)
    lat_1d = np.arange(-90, 91, 1)
    lon_1d = np.arange(-180, 181, 1)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Perturb them slightly to make it "curvilinear"
    lat_2d = lat_2d + 0.1 * np.sin(np.radians(lon_2d))

    data = da.random.random((len(lat_1d), len(lon_1d)), chunks=(91, 181))
    da_source = xr.DataArray(
        data,
        coords={"lat": (("y", "x"), lat_2d), "lon": (("y", "x"), lon_2d)},
        dims=["y", "x"],
        name="test_data"
    )

    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(-180, 181, 10)
    ds_target = xr.Dataset(
        coords={"lat": target_lat, "lon": target_lon}
    )
    return da_source, ds_target

def test_rectilinear_stat_consolidation():
    da_source, ds_target = create_rectilinear_data()
    regridder = RectilinearRegridder(da_source, ds_target)

    # Test stat method
    result = regridder.stat(method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(ds_target.lat), len(ds_target.lon))
    assert "history" in result.attrs

    # Test info
    info = regridder.info()
    assert info["type"] == "RectilinearRegridder"
    assert info["grid_type"] == "rectilinear"
    assert "source" in info
    assert "target" in info

def test_curvilinear_stat_consolidation():
    da_source, ds_target = create_curvilinear_data()
    regridder = CurvilinearRegridder(da_source, ds_target)

    # Test stat method
    result = regridder.stat(method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(ds_target.lat), len(ds_target.lon))
    assert "history" in result.attrs

    # Test info
    info = regridder.info()
    assert info["type"] == "CurvilinearRegridder"
    assert info["grid_type"] == "curvilinear"
    assert "source" in info
    assert "target" in info

def test_categorical_consolidation():
    da_source, ds_target = create_rectilinear_data()
    # Convert to integer for categorical regridding
    da_source = (da_source * 10).astype(int)
    regridder = RectilinearRegridder(da_source, ds_target)

    values = np.arange(11)

    # Test most_common
    result_most = regridder.most_common(values=values)
    assert isinstance(result_most, xr.DataArray)

    # Test least_common
    result_least = regridder.least_common(values=values)
    assert isinstance(result_least, xr.DataArray)

    assert "most_common" in result_most.attrs["history"]
    assert "least_common" in result_least.attrs["history"]

def test_data_agnostic_info():
    _, ds_target = create_rectilinear_data()
    regridder = RectilinearRegridder(source_data=None, target_grid=ds_target)

    info = regridder.info()
    assert info["source"] == {}
    assert info["source_dims"] == {}
    assert "target" in info
