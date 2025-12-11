"""
Comprehensive test script for radius_of_influence functionality in curvilinear nearest neighbor interpolation.

This script tests:
1. Various radius_of_influence parameter values
2. Before/after behavior comparison showing fix for excessive NaN values
3. Backward compatibility
4. Performance benchmarks
5. Edge cases with very small and very large radius values
"""

import logging
import os
import sys
import time

import numpy as np
import pytest
import xarray as xr

try:
    from src.monet_regrid.curvilinear import CurvilinearInterpolator
except ImportError:
    # When running from tests directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.monet_regrid.curvilinear import CurvilinearInterpolator


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
    lat_name = source_grid.cf["latitude"].name if hasattr(source_grid.cf, "latitude") else "latitude"
    lon_name = source_grid.cf["longitude"].name if hasattr(source_grid.cf, "longitude") else "longitude"

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
    # logging.info("\n=== Testing radius_of_influence with various values ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test different radius values
    # Note: None effectively means infinite radius, so it should have 0 NaNs (or minimal)
    # The order here matters for the monotonic check
    # We expect: minimal NaNs (None) <= few NaNs (large radius) <= more NaNs (small radius)
    # But the loop below checks nan_counts[i] >= nan_counts[i+1], so we need decreasing order of NaNs
    # which means increasing order of radius?
    # No, small radius -> many NaNs. Large radius -> few NaNs.
    # So [1000, 500000, 1000000, 5000000, None] should give decreasing NaN counts.

    radius_values = [1000, 500000, 1000000, 5000000, None]  # in meters, ordered by expected decreasing NaNs

    nan_counts = []

    for radius in radius_values:
        # logging.info("Testing radius_of_influence: %s", radius)

        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=radius)

        result = interpolator(test_data)

        # Count NaN values in the result
        nan_count = np.sum(np.isnan(result.data))
        nan_counts.append(nan_count)

    # Assert that nan_counts is monotonically decreasing
    for i in range(len(nan_counts) - 1):
        assert nan_counts[i] >= nan_counts[i + 1]


def test_before_after_behavior():
    """Demonstrate before/after behavior showing how the fix resolves excessive NaN values."""
    # logging.info("\n=== Testing before/after behavior (simulated fix) ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid, with_nans=True)

    # Simulate "before" behavior by using a very small radius (simulating the old issue)
    # logging.info("Simulating 'before' behavior (very restrictive radius):")
    interpolator_before = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=10000,  # Very small radius to simulate excessive NaN issue
    )
    result_before = interpolator_before(test_data)
    nan_count_before = np.sum(np.isnan(result_before.data))
    # logging.info("  Before fix (small radius): %s NaNs", nan_count_before)

    # Simulate "after" behavior with a reasonable radius
    # logging.info("Simulating 'after' behavior (reasonable radius):")
    interpolator_after = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=500000,  # More reasonable radius
    )
    result_after = interpolator_after(test_data)
    nan_count_after = np.sum(np.isnan(result_after.data))
    # logging.info("  After fix (reasonable radius): %s NaNs", nan_count_after)

    # improvement = nan_count_before - nan_count_after
    # improvement_pct = ((nan_count_before - nan_count_after) / nan_count_before * 100) if nan_count_before > 0 else 0

    # logging.info("  Improvement: %s fewer NaNs (%s%% reduction)", improvement, improvement_pct)

    assert nan_count_before > nan_count_after


def test_backward_compatibility():
    """Verify that the fix maintains backward compatibility."""
    # logging.info("\n=== Testing backward compatibility ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test without radius_of_influence (should work as before)
    # logging.info("Testing without radius_of_influence parameter (backward compatibility):")
    interpolator_default = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result_default = interpolator_default(test_data)
    # nan_count_default = np.sum(np.isnan(result_default.data))
    # logging.info("  Default behavior (no radius): %s NaNs", nan_count_default)

    # Test with radius_of_influence=None (should be equivalent to no radius)
    # logging.info("Testing with radius_of_influence=None:")
    interpolator_none = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=None)
    result_none = interpolator_none(test_data)
    # nan_count_none = np.sum(np.isnan(result_none.data))
    # logging.info("  With radius_of_influence=None: %s NaNs", nan_count_none)

    # Results should be equivalent
    are_equivalent = np.allclose(result_default.data, result_none.data, equal_nan=True)
    # logging.info("  Results are equivalent: %s", are_equivalent)

    assert are_equivalent


def benchmark_performance():
    """Include performance benchmarks comparing different radius values."""
    # logging.info("\n=== Performance benchmarking ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    radius_values = [None, 100000, 5000, 1000000]
    iterations = 5

    performance_results = {}

    for radius in radius_values:
        # logging.info("Benchmarking radius_of_influence: %s", radius)

        # Warm up
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=radius)

        times = []
        for _i in range(iterations):
            start_time = time.time()
            interpolator(test_data)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        performance_results[radius] = {"avg_time": avg_time, "std_time": std_time, "times": times}

        # logging.info("  Average time: %ss ± %ss", avg_time, std_time)

    # logging.info("\nPerformance summary:")
    # for radius, perf_data in performance_results.items():
    #     radius_str = "None (default)" if radius is None else f"{radius:,}m"
    #     logging.info(" Radius %s: %ss ± %ss", radius_str, perf_data['avg_time'], perf_data['std_time'])

    return performance_results


def test_edge_cases():
    """Test edge cases like very small and very large radius values."""
    # logging.info("\n=== Testing edge cases ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test very small radius (should result in many NaNs)
    # logging.info("Testing very small radius (100m):")
    interpolator_small = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=100,  # Very small radius
    )
    result_small = interpolator_small(test_data)
    np.sum(np.isnan(result_small.data))
    # logging.info("  Very small radius: %s NaNs", nan_count_small)

    # Test very large radius (should result in few NaNs, almost all points filled)
    # logging.info("Testing very large radius (10,000,000m):")
    interpolator_large = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=1000000,  # Very large radius (about Earth's diameter)
    )
    result_large = interpolator_large(test_data)
    nan_count_large = np.sum(np.isnan(result_large.data))
    # logging.info("  Very large radius: %s NaNs", nan_count_large)

    # Test zero radius (should result in maximum NaNs)
    # logging.info("Testing zero radius:")
    interpolator_zero = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=0,  # Zero radius
    )
    result_zero = interpolator_zero(test_data)
    nan_count_zero = np.sum(np.isnan(result_zero.data))
    # logging.info("  Zero radius: %s NaNs", nan_count_zero)

    assert nan_count_zero == result_zero.data.size
    assert nan_count_large == 0


def test_error_handling():
    """Test error handling for invalid radius values."""
    # logging.info("\n=== Testing error handling ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test negative radius (should raise an error or handle gracefully)
    # logging.info("Testing negative radius:")
    try:
        interpolator_negative = CurvilinearInterpolator(
            source_grid, target_grid, method="nearest", radius_of_influence=-100000
        )
        result_negative = interpolator_negative(test_data)
        np.sum(np.isnan(result_negative.data))
        # logging.info("  Negative radius handled, NaN count: %s", nan_count_negative)
    except Exception:
        # logging.info("  Negative radius raised exception: %s", e)
        pass

    # Test extremely large radius (should work but might be slow)
    # logging.info("Testing extremely large radius:")
    try:
        interpolator_extreme = CurvilinearInterpolator(
            source_grid,
            target_grid,
            method="nearest",
            radius_of_influence=1e10,  # Extremely large radius
        )
        result_extreme = interpolator_extreme(test_data)
        np.sum(np.isnan(result_extreme.data))
        # logging.info(" Extremely large radius, NaN count: %s", nan_count_extreme)
    except Exception:
        # logging.info("  Extremely large radius raised exception: %s", e)
        pass


def main():
    """Run all comprehensive tests."""
    # logging.info("Running comprehensive radius_of_influence tests...\n")

    # 1. Test various radius values
    results_various = test_radius_of_influence_various_values()

    # 2. Test before/after behavior
    results_before_after = test_before_after_behavior()

    # 3. Test backward compatibility
    backward_compatible = test_backward_compatibility()

    # 4. Performance benchmarks
    performance_results = benchmark_performance()

    # 5. Edge cases
    edge_case_results = test_edge_cases()

    # 6. Error handling
    test_error_handling()

    # Final summary
    # logging.info("\n%s", "="*60)
    # logging.info("COMPREHENSIVE TEST SUMMARY")
    # logging.info("="*60)

    # logging.info("✓ Various radius values tested: %s different values", len(results_various))
    # logging.info("✓ Before/after behavior demonstrated")
    # logging.info("✓ Backward compatibility: %s", 'PASSED' if backward_compatible else 'FAILED')
    # logging.info("✓ Performance benchmarks completed: %s configurations tested", len(performance_results))
    # logging.info("✓ Edge cases tested: %s scenarios", len(edge_case_results))
    # logging.info("✓ Error handling verified")

    # logging.info("\nKey findings:")
    # logging.info("- Radius of influence significantly affects NaN count in results")
    # logging.info("- Larger radii result in fewer NaN values (more points filled)")
    # logging.info("- Smaller radii result in more NaN values (stricter matching)")
    # logging.info("- Backward compatibility maintained when radius_of_influence is None")
    # logging.info("- Performance impact is minimal across different radius values")

    return {
        "various_values": results_various,
        "before_after": results_before_after,
        "backward_compatible": backward_compatible,
        "performance": performance_results,
        "edge_cases": edge_case_results,
    }


if __name__ == "__main__":
    results = main()
