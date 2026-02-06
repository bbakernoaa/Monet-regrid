from __future__ import annotations

import numpy as np
import xarray as xr

from monet_regrid.methods.interp import _interp_regrid_fast, interp_regrid


def test_interp_fast_extra_dims() -> None:
    """Test that the fast path handles extra dimensions correctly."""
    # Source grid
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    time = np.arange(5)

    data = np.random.rand(5, 10, 20)
    da = xr.DataArray(data, dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}, name="test_data")

    # Target grid
    lat_new = np.linspace(-90, 90, 15)
    lon_new = np.linspace(-180, 180, 25)
    target_ds = xr.Dataset(coords={"lat": lat_new, "lon": lon_new})

    # Fast path result
    res_fast = interp_regrid(da, target_ds, method="linear")

    # Slow path (fallback) result
    res_slow = da.interp(lat=lat_new, lon=lon_new, method="linear")

    # Assertions
    xr.testing.assert_allclose(res_fast, res_slow)
    assert res_fast.dims == ("time", "lat", "lon")
    assert "history" in res_fast.attrs
    assert "monet_regrid.methods.interp.interp_regrid" in res_fast.attrs["history"]


def test_interp_fast_dataset() -> None:
    """Test that the fast path works for Datasets."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)

    ds = xr.Dataset(
        {
            "a": (("lat", "lon"), np.random.rand(10, 20)),
            "b": (("lat", "lon"), np.random.rand(10, 20)),
        },
        coords={"lat": lat, "lon": lon},
    )

    target_ds = xr.Dataset(coords={"lat": np.linspace(-90, 90, 15), "lon": np.linspace(-180, 180, 25)})

    res = interp_regrid(ds, target_ds, method="nearest")

    assert isinstance(res, xr.Dataset)
    assert "a" in res.data_vars
    assert "b" in res.data_vars
    assert res.a.shape == (15, 25)
    assert "history" in res.attrs


def test_interp_fast_consistency() -> None:
    """Verify _interp_regrid_fast directly and compare with xr.interp."""
    lat = np.linspace(0, 10, 5)
    lon = np.linspace(0, 10, 5)
    data = np.arange(25).reshape(5, 5).astype(float)

    da = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    target_ds = xr.Dataset(coords={"lat": [2.5, 7.5], "lon": [2.5, 7.5]})

    res_fast = _interp_regrid_fast(da, target_ds, "linear", ["lat", "lon"])
    res_xr = da.interp(lat=[2.5, 7.5], lon=[2.5, 7.5], method="linear")

    xr.testing.assert_allclose(res_fast, res_xr)


def test_interp_fast_multiple_extra_dims() -> None:
    """Test with more than one extra dimension."""
    # (lev, time, lat, lon)
    da = xr.DataArray(
        np.random.rand(2, 3, 4, 5),
        dims=["lev", "time", "lat", "lon"],
        coords={"lev": [1, 2], "time": [10, 20, 30], "lat": np.linspace(0, 1, 4), "lon": np.linspace(0, 1, 5)},
    )

    target_ds = xr.Dataset(coords={"lat": [0.2, 0.8], "lon": [0.3, 0.7]})

    res = interp_regrid(da, target_ds, method="linear")

    assert res.dims == ("lev", "time", "lat", "lon")
    assert res.shape == (2, 3, 2, 2)

    # Compare with slow path
    expected = da.interp(lat=[0.2, 0.8], lon=[0.3, 0.7], method="linear")
    xr.testing.assert_allclose(res, expected)


def test_interp_auxiliary_coordinates() -> None:
    """Verify that auxiliary coordinates are preserved or interpolated."""
    # Source grid
    lat = np.arange(10)
    lon = np.arange(10)

    # Auxiliary coordinate depending on lat/lon
    alt = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    # Auxiliary coordinate NOT depending on lat/lon
    fixed_val = xr.DataArray(42, name="fixed_val")

    da = xr.DataArray(
        np.random.rand(10, 10),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon, "alt": alt, "fixed_val": fixed_val},
        name="test_data",
    )

    target_ds = xr.Dataset(coords={"lat": np.arange(0, 10, 2), "lon": np.arange(0, 10, 2)})

    res = interp_regrid(da, target_ds, method="linear")

    # Verify dimensions
    assert res.shape == (5, 5)

    # Verify auxiliary coordinates
    assert "alt" in res.coords
    assert res.alt.shape == (5, 5)  # Should have been interpolated
    assert "fixed_val" in res.coords
    assert res.fixed_val == 42  # Should have been preserved

    # Compare with slow path
    expected = da.interp(lat=target_ds.lat, lon=target_ds.lon, method="linear")
    xr.testing.assert_allclose(res, expected)


def test_interp_dataset_coordinates() -> None:
    """Verify that Dataset coordinates (including non-dimension ones) are handled."""
    lat = np.arange(10)
    lon = np.arange(10)
    alt = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    ds = xr.Dataset({"temp": (("lat", "lon"), np.random.rand(10, 10))}, coords={"lat": lat, "lon": lon, "alt": alt})

    target_ds = xr.Dataset(coords={"lat": np.arange(0, 10, 2), "lon": np.arange(0, 10, 2)})

    res = interp_regrid(ds, target_ds, method="linear")

    assert "alt" in res.coords
    assert res.alt.shape == (5, 5)

    # Compare with slow path
    expected = ds.interp(lat=target_ds.lat, lon=target_ds.lon, method="linear")
    xr.testing.assert_allclose(res, expected)
