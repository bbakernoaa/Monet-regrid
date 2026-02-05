"""Unit tests for curvilinear bounds broadcasting logic."""

import numpy as np
import pytest
import xarray as xr


def test_curvilinear_regridder_rectilinear_target_bounds():
    """Test that CurvilinearRegridder handles rectilinear target grids with 1D bounds."""
    # 1. Setup source curvilinear grid
    lon = np.linspace(0, 10, 5)
    lat = np.linspace(0, 10, 5)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # 2D Bounds for source (continuous grid)
    # dx = 2.5
    lon_b = np.stack([lon2d - 1.25, lon2d + 1.25, lon2d + 1.25, lon2d - 1.25], axis=-1)
    lat_b = np.stack([lat2d - 1.25, lat2d - 1.25, lat2d + 1.25, lat2d + 1.25], axis=-1)

    ds_source = xr.Dataset(
        data_vars={"data": (("y", "x"), np.ones((5, 5)))},
        coords={
            "lat": (("y", "x"), lat2d, {"bounds": "lat_b"}),
            "lon": (("y", "x"), lon2d, {"bounds": "lon_b"}),
            "lat_b": (("y", "x", "nv"), lat_b),
            "lon_b": (("y", "x", "nv"), lon_b),
        },
    )

    # 2. Setup target rectilinear grid with 1D bounds
    t_lon = np.linspace(2, 8, 3)
    t_lat = np.linspace(2, 8, 3)

    t_lon_b = np.stack([t_lon - 1, t_lon + 1], axis=-1)
    t_lat_b = np.stack([t_lat - 1, t_lat + 1], axis=-1)

    ds_target = xr.Dataset(
        coords={
            "lat": (("lat",), t_lat, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("lon",), t_lon, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("lat", "nv"), t_lat_b),
            "lon_b": (("lon", "nv"), t_lon_b),
        }
    )

    # 3. Execute regridding
    # This should use CurvilinearRegridder because ds_source has 2D coords
    # and it should now correctly broadcast the 1D target bounds.
    regridded = ds_source.regrid.conservative(ds_target)

    # 4. Verify
    assert "data" in regridded.data_vars
    assert regridded.data.shape == (3, 3)
    assert not np.isnan(regridded.data.values).any()
    # Since source is all ones, regridded should be close to 1 (if normalization is correct)
    np.testing.assert_allclose(regridded.data.values, 1.0, rtol=1e-5)


def test_curvilinear_regridder_mismatched_target_bounds():
    """Test error handling when target bounds are missing or incorrect."""
    lon = np.linspace(0, 10, 5)
    lat = np.linspace(0, 10, 5)
    lon2d, lat2d = np.meshgrid(lon, lat)

    ds_source = xr.Dataset(
        data_vars={"data": (("y", "x"), np.ones((5, 5)))},
        coords={
            "lat": (("y", "x"), lat2d, {"bounds": "lat_b"}),
            "lon": (("y", "x"), lon2d, {"bounds": "lon_b"}),
            "lat_b": (("y", "x", "nv"), np.random.rand(5, 5, 4)),
            "lon_b": (("y", "x", "nv"), np.random.rand(5, 5, 4)),
        },
    )

    # Target missing bounds
    ds_target_no_bnds = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(2, 8, 3)),
            "lon": (("lon",), np.linspace(2, 8, 3)),
        }
    )

    with pytest.raises(ValueError, match="Conservative regridding requires explicit bounds"):
        ds_source.regrid.conservative(ds_target_no_bnds)
