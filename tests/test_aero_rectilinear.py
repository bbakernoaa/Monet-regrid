import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.methods.flox_reduce import statistic_reduce
from monet_regrid.methods.interp import interp_regrid


def test_interp_regrid_provenance():
    """Verify that interp_regrid adds history for provenance."""
    ds = xr.Dataset({"a": (("lat", "lon"), np.random.rand(10, 10))}, coords={"lat": np.arange(10), "lon": np.arange(10)})
    target = xr.Dataset(coords={"lat": np.linspace(0, 9, 20), "lon": np.linspace(0, 9, 20)})

    res = interp_regrid(ds, target, method="linear")

    assert "history" in res.attrs
    assert "Interpolated using monet_regrid.methods.interp.interp_regrid" in res.attrs["history"]


def test_statistic_reduce_provenance():
    """Verify that statistic_reduce adds history for provenance."""
    ds = xr.Dataset({"a": (("lat", "lon"), np.random.rand(10, 10))}, coords={"lat": np.arange(10), "lon": np.arange(10)})
    target = xr.Dataset(coords={"lat": [2, 5, 8], "lon": [2, 5, 8]})

    res = statistic_reduce(ds, target, time_dim=None, method="mean")

    assert "history" in res.attrs
    assert "Reduced using monet_regrid.methods.flox_reduce.statistic_reduce" in res.attrs["history"]


def test_interp_regrid_lazy():
    """Verify that interp_regrid preserves dask arrays (lazy evaluation)."""
    data = da.random.random((10, 10), chunks=(5, 5))
    ds = xr.Dataset({"a": (("lat", "lon"), data)}, coords={"lat": np.arange(10), "lon": np.arange(10)})
    target = xr.Dataset(coords={"lat": np.linspace(0, 9, 20), "lon": np.linspace(0, 9, 20)})

    res = interp_regrid(ds, target, method="linear")

    # Check if it's still a dask array
    assert isinstance(res.a.data, da.Array)


def test_statistic_reduce_lazy():
    """Verify that statistic_reduce preserves dask arrays."""
    data = da.random.random((10, 10), chunks=(5, 5))
    ds = xr.Dataset({"a": (("lat", "lon"), data)}, coords={"lat": np.arange(10), "lon": np.arange(10)})
    target = xr.Dataset(coords={"lat": [2, 5, 8], "lon": [2, 5, 8]})

    res = statistic_reduce(ds, target, time_dim=None, method="mean")

    assert isinstance(res.a.data, da.Array)
