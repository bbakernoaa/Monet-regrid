"""Test for radius_of_influence parameter in CurvilinearInterpolator."""

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from xarray_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


def test_curvilinear_nearest_with_radius_of_influence():
    """Test that radius_of_influence parameter works correctly in curvilinear nearest neighbor interpolation."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

    target_x, target_y = np.meshgrid(np.linspace(0, 4, 3), np.linspace(0, 5, 4))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.random.rand(6, 5)  # (y, x)
    test_data = xr.DataArray(data_values, dims=["y", "x"], coords={"y": range(6), "x": range(5)})

    # Test nearest neighbor interpolation with a large radius of influence
    # This should produce results with fewer NaNs compared to default behavior
    interpolator_large_radius = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", radius_of_influence=1e6
    )
    result_large_radius = interpolator_large_radius(test_data)

    # Test nearest neighbor interpolation with a small radius of influence
    # This should produce results with more NaNs
    interpolator_small_radius = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", radius_of_influence=1e3
    )
    result_small_radius = interpolator_small_radius(test_data)

    # Verify that the large radius produces fewer NaNs than the small radius
    # (in most cases, unless the small radius is already sufficient for all points)
    large_radius_nans = np.isnan(result_large_radius).sum()
    small_radius_nans = np.isnan(result_small_radius).sum()

    # The large radius should not have more NaNs than the small radius
    assert large_radius_nans <= small_radius_nans, (
        f"Large radius should not have more NaNs than small radius: {large_radius_nans} > {small_radius_nans}"
    )

    # Verify that the radius of influence parameter is correctly stored
    assert interpolator_large_radius.radius_of_influence == 1e6
    assert interpolator_small_radius.radius_of_influence == 1e3

    # Test with a very large radius - should produce no NaNs (or very few) if all target points are within range
    interpolator_very_large_radius = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", radius_of_influence=1e10
    )
    result_very_large_radius = interpolator_very_large_radius(test_data)

    # The very large radius should have the same or fewer NaNs than the small radius
    very_large_radius_nans = np.isnan(result_very_large_radius).sum()
    assert very_large_radius_nans <= small_radius_nans, (
        f"Very large radius should not have more NaNs than small radius: {very_large_radius_nans} > {small_radius_nans}"
    )


if __name__ == "__main__":
    test_curvilinear_nearest_with_radius_of_influence()
    print("All tests passed!")
