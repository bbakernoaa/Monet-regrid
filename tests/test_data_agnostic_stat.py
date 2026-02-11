import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def test_rectilinear_data_agnostic_stat():
    """Verify that RectilinearRegridder supports data-agnostic stat calls."""
    # Create target grid
    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(0, 360, 10)
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    target.lat.attrs = {"units": "degrees_north"}
    target.lon.attrs = {"units": "degrees_east"}

    # Initialize data-agnostic regridder
    regridder = RectilinearRegridder(None, target)

    # Create source data
    lat = np.arange(-90, 91, 2)
    lon = np.arange(0, 360, 2)
    data = np.ones((len(lat), len(lon)))
    source_da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="test")
    source_da.lat.attrs = {"units": "degrees_north"}
    source_da.lon.attrs = {"units": "degrees_east"}

    # Apply regridder with data
    result = regridder.stat(method="mean", data=source_da)

    assert result.shape == (len(target_lat), len(target_lon))
    np.testing.assert_allclose(result.values, 1.0)
    assert "history" in result.attrs
    assert "statistic_reduce" in result.attrs["history"]


def test_curvilinear_data_agnostic_stat():
    """Verify that CurvilinearRegridder supports data-agnostic stat calls."""
    # Create target grid (rectilinear target is common)
    target_lat = np.arange(-90, 91, 20)
    target_lon = np.arange(0, 360, 20)
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    target.lat.attrs = {"units": "degrees_north"}
    target.lon.attrs = {"units": "degrees_east"}

    # Initialize data-agnostic regridder
    regridder = CurvilinearRegridder(None, target)

    # Create source data (curvilinear)
    lon_2d, lat_2d = np.meshgrid(np.arange(0, 360, 10), np.arange(-90, 91, 10))
    source_da = xr.DataArray(
        np.ones(lat_2d.shape),
        coords={"latitude": (("y", "x"), lat_2d), "longitude": (("y", "x"), lon_2d)},
        dims=["y", "x"],
        name="test",
    )
    source_da.latitude.attrs = {"units": "degrees_north"}
    source_da.longitude.attrs = {"units": "degrees_east"}

    # Apply regridder with data
    result = regridder.stat(method="mean", data=source_da)

    assert result.shape == (len(target_lat), len(target_lon))
    np.testing.assert_allclose(result.values, 1.0)
    assert "history" in result.attrs
    assert "statistic_reduce" in result.attrs["history"]


def test_most_common_data_agnostic():
    """Verify that most_common supports data-agnostic calls."""
    target_lat = np.arange(-90, 91, 30)
    target_lon = np.arange(0, 360, 30)
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    target.lat.attrs = {"units": "degrees_north"}
    target.lon.attrs = {"units": "degrees_east"}

    regridder = RectilinearRegridder(None, target)

    lat = np.arange(-90, 91, 5)
    lon = np.arange(0, 360, 5)
    data = np.zeros((len(lat), len(lon)), dtype=int)
    data[0:2, 0:2] = 1
    source_da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="test")
    source_da.lat.attrs = {"units": "degrees_north"}
    source_da.lon.attrs = {"units": "degrees_east"}

    result = regridder.most_common(values=np.array([0, 1]), data=source_da)

    assert result.dtype == int
    assert "history" in result.attrs
    assert "most_common" in result.attrs["history"]
