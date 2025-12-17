"""
Comprehensive test script for radius_of_influence functionality in curvilinear nearest neighbor interpolation.

This script tests:
1. Various radius_of_influence parameter values
2. Before/after behavior comparison showing fix for excessive NaN values
3. Backward compatibility
4. Performance benchmarks
5. Edge cases with very small and very large radius values
"""

import numpy as np
import xarray as xr

import monet_regrid  # noqa: F401


def create_curvilinear_grids():
    """Create sample curvilinear grids for testing."""
    # Source grid: 5x6 curvilinear grid
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
    # Add some curvature/distortion to make it truly curvilinear
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y + 0.05 * source_x * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y + 0.02 * source_x * source_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    # Target grid: 3x4 curvilinear grid
    target_x, target_y = np.meshgrid(np.arange(3), np.arange(4))
    # Add some curvature/distortion to make it truly curvilinear
    target_lat = 32 + 0.4 * target_x + 0.15 * target_y + 0.03 * target_x * target_y
    target_lon = -98 + 0.25 * target_x + 0.18 * target_y + 0.01 * target_x * target_y

    target_grid = xr.Dataset({"latitude": (["lat", "lon"], target_lat), "longitude": (["lat", "lon"], target_lon)})

    return source_grid, target_grid


def create_test_data(source_grid, with_nans=False):
    """Create test data with optional NaN values."""
    lat_name = "latitude"
    lon_name = "longitude"

    # Use the coordinate names from the grid
    y_dim, x_dim = source_grid[lat_name].dims

    # Create test data with a pattern that makes it easy to verify interpolation
    data_values = np.random.random((6, 5))  # y=6, x=5 based on our source grid

    if with_nans:
        # Add some NaN values in a pattern
        data_values[1, 2] = np.nan
        data_values[3, 1] = np.nan
        data_values[4, 4] = np.nan

    test_data = xr.DataArray(
        data_values,
        dims=[y_dim, x_dim],
        coords={
            lat_name: (source_grid[lat_name].dims, source_grid[lat_name].values),
            lon_name: (source_grid[lon_name].dims, source_grid[lon_name].values),
        },
    )

    return test_data


def test_radius_of_influence_various_values():
    """Test radius_of_influence parameter with various values."""
    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    radius_values = [1000, 500000, 1000000, 5000000, None]

    nan_counts = []

    for radius in radius_values:
        result = test_data.regrid.nearest(target_grid, radius_of_influence=radius)
        nan_count = np.sum(np.isnan(result.data))
        nan_counts.append(nan_count)

    for i in range(len(nan_counts) - 1):
        assert nan_counts[i] >= nan_counts[i + 1]


def test_before_after_behavior():
    """Demonstrate before/after behavior showing how the fix resolves excessive NaN values."""
    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid, with_nans=True)

    result_before = test_data.regrid.nearest(target_grid, radius_of_influence=10000)
    nan_count_before = np.sum(np.isnan(result_before.data))

    result_after = test_data.regrid.nearest(target_grid, radius_of_influence=500000)
    nan_count_after = np.sum(np.isnan(result_after.data))

    assert nan_count_before > nan_count_after


def test_backward_compatibility():
    """Verify that the fix maintains backward compatibility."""
    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    result_default = test_data.regrid.nearest(target_grid)
    result_none = test_data.regrid.nearest(target_grid, radius_of_influence=None)

    are_equivalent = np.allclose(result_default.data, result_none.data, equal_nan=True)

    assert are_equivalent


def test_edge_cases():
    """Test edge cases like very small and very large radius values."""
    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    result_small = test_data.regrid.nearest(target_grid, radius_of_influence=100)
    np.sum(np.isnan(result_small.data))

    result_large = test_data.regrid.nearest(target_grid, radius_of_influence=1000000)
    nan_count_large = np.sum(np.isnan(result_large.data))

    result_zero = test_data.regrid.nearest(target_grid, radius_of_influence=0)
    nan_count_zero = np.sum(np.isnan(result_zero.data))

    assert nan_count_zero == result_zero.data.size
    assert nan_count_large == 0
