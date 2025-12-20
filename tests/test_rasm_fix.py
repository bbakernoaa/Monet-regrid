"""
Test case for a specific issue with RASM data where curvilinear regridding
produced incorrect results. This test ensures the fix is effective.
"""

import numpy as np
import xarray as xr

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from monet_regrid import CurvilinearRegridder
# New import: from monet_regrid import CurvilinearRegridder


def generate_rasm_like_grids():
    """Generate simplified grids that mimic the RASM data structure."""
    # Source grid (curvilinear, high-resolution)
    ny_source, nx_source = 10, 10
    lon = np.linspace(-100, -80, nx_source)
    lat = np.linspace(30, 40, ny_source)
    lon2d, lat2d = np.meshgrid(lon, lat)
    # Add some non-linearity
    lat2d += np.sin(np.deg2rad(lon2d)) * 0.5
    source_grid = xr.Dataset(
        {"latitude": (("y", "x"), lat2d), "longitude": (("y", "x"), lon2d)},
        coords={"x": np.arange(nx_source), "y": np.arange(ny_source)},
    )

    # Target grid (rectilinear, lower-resolution)
    target_grid = xr.Dataset(
        coords={"latitude": np.linspace(32, 38, 5), "longitude": np.linspace(-98, -82, 6)}
    )
    return source_grid, target_grid


def test_rasm_regridding_fix():
    """Test that regridding from a RASM-like grid to a rectilinear grid works correctly."""
    source_grid, target_grid = generate_rasm_like_grids()

    # Create a DataArray with a smooth gradient for easy verification
    # Using longitude as a proxy for the gradient
    data_values = source_grid["longitude"] ** 2
    source_da = xr.DataArray(
        data_values.values,
        dims=("y", "x"),
        coords={
            "latitude": (("y", "x"), source_grid.latitude.values),
            "longitude": (("y", "x"), source_grid.longitude.values),
        },
    )

    # Perform regridding
    result_da = source_da.regrid.linear(target_grid)

    # Verification
    # 1. The result should have the shape of the target grid.
    assert result_da.shape == (5, 6)

    # 2. Check that the coordinates of the result match the target grid.
    np.testing.assert_allclose(result_da.coords["latitude"], target_grid.latitude.values)
    np.testing.assert_allclose(result_da.coords["longitude"], target_grid.longitude.values)

    # 3. Check for a reasonable range of values. The regridded values
    #    should be within the range of the source data.
    min_source = source_da.min().item()
    max_source = source_da.max().item()
    assert result_da.min().item() >= min_source - 1e-5
    assert result_da.max().item() <= max_source + 1e-5

    # 4. Check for smoothness - there should not be any abrupt jumps or NaNs.
    assert np.all(np.isfinite(result_da.values))
    grad_lon = np.gradient(result_da.values, axis=1)
    grad_lat = np.gradient(result_da.values, axis=0)
    assert np.all(np.abs(grad_lon) < 700)  # Should be smooth
    assert np.all(np.abs(grad_lat) < 700)  # Should be smooth
