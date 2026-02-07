import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.methods.interp import interp_regrid


def test_interp_regrid_fast_is_lazy():
    """Verify that interp_regrid with fast path is lazy for Dask-backed data."""
    # Create large synthetic Dask data
    nlat, nlon = 100, 200
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)

    data = da.random.random((nlat, nlon), chunks=(50, 50))
    da_src = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="test_data")

    # Target grid
    target_lat = np.linspace(-90, 90, 50)
    target_lon = np.linspace(-180, 180, 100)
    target_ds = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # Perform regridding (should be fast path since it's rectilinear)
    # The fast path now supports Dask!
    result = interp_regrid(da_src, target_ds, method="linear")

    # Verify it is still a Dask array
    assert isinstance(result.data, da.Array)

    # Verify it hasn't been computed yet
    # (Actually it's hard to verify "hasn't been computed" without mocks,
    # but we can check if it has chunks)
    assert result.chunks is not None

    # Compute and verify results
    computed_result = result.compute()
    assert computed_result.shape == (50, 100)
    assert not np.any(np.isnan(computed_result))


def test_interp_regrid_fast_multidim_lazy():
    """Verify that interp_regrid fast path works lazily with extra dimensions."""
    ntime, nlat, nlon = 5, 100, 200
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)
    time = np.arange(ntime)

    data = da.random.random((ntime, nlat, nlon), chunks=(1, 50, 50))
    da_src = xr.DataArray(data, dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}, name="test_data")

    target_ds = xr.Dataset(coords={"lat": np.linspace(-90, 90, 50), "lon": np.linspace(-180, 180, 100)})

    result = interp_regrid(da_src, target_ds, method="nearest")

    assert isinstance(result.data, da.Array)
    assert result.chunks is not None
    assert result.dims == ("time", "lat", "lon")

    computed = result.compute()
    assert computed.shape == (5, 50, 100)
