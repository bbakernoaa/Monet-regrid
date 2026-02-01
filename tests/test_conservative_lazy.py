import dask.array as da
import numpy as np
import xarray as xr

import monet_regrid  # noqa: F401


def test_conservative_regrid_is_lazy():
    """Verify that conservative regridding remains lazy for Dask-backed data.

    This test ensures that the conservative regridding implementation follows
    the Aero Protocol by maintaining laziness when operating on Dask arrays.
    """
    # 1. Create a Dask-backed source dataset (10x10)
    data = da.random.random((10, 10), chunks=(5, 5))
    ds_source = xr.Dataset(
        {"var": (("lat", "lon"), data)},
        coords={
            "lat": (["lat"], np.arange(-45, 46, 10), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0, 100, 10), {"units": "degrees_east"}),
        },
    )

    # 2. Create a target grid (5x5)
    ds_target = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-40, 41, 20), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(10, 91, 20), {"units": "degrees_east"}),
        }
    )

    # 3. Apply conservative regridding
    ds_regrid = ds_source.regrid.conservative(ds_target)

    # 4. Assert that the result is still a Dask array
    assert isinstance(ds_regrid["var"].data, da.Array), "Result should be a Dask array"

    # 5. Assert that the history attribute was updated
    assert "history" in ds_regrid.attrs
    assert "Regridded using conservative method" in ds_regrid.attrs["history"]

    # 6. Compute and verify shape
    result = ds_regrid.compute()
    assert result["var"].shape == (5, 5), f"Expected shape (5, 5), got {result['var'].shape}"

    # 7. Verify no NaN values if input was full
    assert not np.isnan(result["var"].values).any(), "Result should not contain NaNs for this input"


def test_conservative_regrid_spherical_correction_laziness():
    """Verify that spherical correction doesn't break laziness."""
    # Create global-like grid to trigger spherical correction
    lats = np.arange(-85, 86, 10)  # 18 points
    lons = np.arange(0, 360, 20)  # 18 points
    data = da.random.random((len(lats), len(lons)), chunks=(9, 9))

    ds_source = xr.Dataset(
        {"var": (("lat", "lon"), data)},
        coords={"lat": (["lat"], lats, {"units": "degrees_north"}), "lon": (["lon"], lons, {"units": "degrees_east"})},
    )

    ds_target = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-80, 81, 20), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(10, 351, 40), {"units": "degrees_east"}),
        }
    )

    ds_regrid = ds_source.regrid.conservative(ds_target)

    assert isinstance(ds_regrid["var"].data, da.Array), "Result should remain a Dask array after spherical correction"

    # Trigger compute to ensure it works
    result = ds_regrid.compute()
    assert result["var"].shape == (9, 9)
