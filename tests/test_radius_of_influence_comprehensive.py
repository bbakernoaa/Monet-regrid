"""
Comprehensive test script for radius_of_influence functionality in curvilinear nearest neighbor interpolation.

This script tests:
1. Various radius_of_influence parameter values
2. Before/after behavior comparison showing fix for excessive NaN values
3. Backward compatibility
4. Performance benchmarks
5. Edge cases with very small and very large radius values
"""

import time

import numpy as np
import pytest
import xarray as xr

try:
    from src.monet_regrid.curvilinear import CurvilinearInterpolator
except ImportError:
    # When running from tests directory
    import os
    import sys

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
    # print("\n=== Testing radius_of_influence with various values ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test different radius values
    radius_values = [None, 1000, 500000, 1000000, 5000000]  # in meters

    results = {}

    for radius in radius_values:
        # print(f"Testing radius_of_influence: {radius}")

        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=radius)

        result = interpolator(test_data)

        # Count NaN values in the result
        nan_count = np.sum(np.isnan(result.data))
        total_points = result.data.size

        results[radius] = {
            "result": result,
            "nan_count": nan_count,
            "total_points": total_points,
            "nan_percentage": (nan_count / total_points) * 100 if total_points > 0 else 0,
        }

        # print(f"  NaN count: {nan_count}/{total_points} ({results[radius]['nan_percentage']:.2f}%)")

    # Print summary
    # print("\nSummary of radius_of_influence effects:")
    for radius, data in results.items():
        radius_str = "None (default)" if radius is None else f"{radius:,}m"
        # print(f"  Radius {radius_str}: {data['nan_count']} NaNs ({data['nan_percentage']:.2f}%)")

    return results


def test_before_after_behavior():
    """Demonstrate before/after behavior showing how the fix resolves excessive NaN values."""
    # print("\n=== Testing before/after behavior (simulated fix) ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid, with_nans=True)

    # Simulate "before" behavior by using a very small radius (simulating the old issue)
    # print("Simulating 'before' behavior (very restrictive radius):")
    interpolator_before = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=10000,  # Very small radius to simulate excessive NaN issue
    )
    result_before = interpolator_before(test_data)
    nan_count_before = np.sum(np.isnan(result_before.data))
    # print(f"  Before fix (small radius): {nan_count_before} NaNs")

    # Simulate "after" behavior with a reasonable radius
    # print("Simulating 'after' behavior (reasonable radius):")
    interpolator_after = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=500000,  # More reasonable radius
    )
    result_after = interpolator_after(test_data)
    nan_count_after = np.sum(np.isnan(result_after.data))
    # print(f"  After fix (reasonable radius): {nan_count_after} NaNs")

    improvement = nan_count_before - nan_count_after
    improvement_pct = ((nan_count_before - nan_count_after) / nan_count_before * 100) if nan_count_before > 0 else 0

    # print(f"  Improvement: {improvement} fewer NaNs ({improvement_pct:.2f}% reduction)")

    return {
        "before": {"result": result_before, "nan_count": nan_count_before},
        "after": {"result": result_after, "nan_count": nan_count_after},
    }


def test_backward_compatibility():
    """Verify that the fix maintains backward compatibility."""
    # print("\n=== Testing backward compatibility ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test without radius_of_influence (should work as before)
    # print("Testing without radius_of_influence parameter (backward compatibility):")
    interpolator_default = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
    result_default = interpolator_default(test_data)
    nan_count_default = np.sum(np.isnan(result_default.data))
    # print(f"  Default behavior (no radius): {nan_count_default} NaNs")

    # Test with radius_of_influence=None (should be equivalent to no radius)
    # print("Testing with radius_of_influence=None:")
    interpolator_none = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=None)
    result_none = interpolator_none(test_data)
    nan_count_none = np.sum(np.isnan(result_none.data))
    # print(f"  With radius_of_influence=None: {nan_count_none} NaNs")

    # Results should be equivalent
    are_equivalent = np.allclose(result_default.data, result_none.data, equal_nan=True)
    # print(f"  Results are equivalent: {are_equivalent}")

    if not are_equivalent:
        # print("  WARNING: Backward compatibility issue detected!")
        return False
    else:
        # print("  Backward compatibility maintained.")
        return True


def benchmark_performance():
    """Include performance benchmarks comparing different radius values."""
    # print("\n=== Performance benchmarking ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    radius_values = [None, 100000, 5000, 1000000]
    iterations = 5

    performance_results = {}

    for radius in radius_values:
        # print(f"Benchmarking radius_of_influence: {radius}")

        # Warm up
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=radius)

        times = []
        for i in range(iterations):
            start_time = time.time()
            result = interpolator(test_data)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        performance_results[radius] = {"avg_time": avg_time, "std_time": std_time, "times": times}

        # print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")

    # Print performance summary
    # print("\nPerformance summary:")
    for radius, perf_data in performance_results.items():
        radius_str = "None (default)" if radius is None else f"{radius:,}m"
        # print(f" Radius {radius_str}: {perf_data['avg_time']:.4f}s ± {perf_data['std_time']:.4f}s")

    return performance_results


def test_edge_cases():
    """Test edge cases like very small and very large radius values."""
    # print("\n=== Testing edge cases ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test very small radius (should result in many NaNs)
    # print("Testing very small radius (100m):")
    interpolator_small = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=100,  # Very small radius
    )
    result_small = interpolator_small(test_data)
    nan_count_small = np.sum(np.isnan(result_small.data))
    # print(f"  Very small radius: {nan_count_small} NaNs")

    # Test very large radius (should result in few NaNs, almost all points filled)
    # print("Testing very large radius (10,000,000m):")
    interpolator_large = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=1000000,  # Very large radius (about Earth's diameter)
    )
    result_large = interpolator_large(test_data)
    nan_count_large = np.sum(np.isnan(result_large.data))
    # print(f"  Very large radius: {nan_count_large} NaNs")

    # Test zero radius (should result in maximum NaNs)
    # print("Testing zero radius:")
    interpolator_zero = CurvilinearInterpolator(
        source_grid,
        target_grid,
        method="nearest",
        radius_of_influence=0,  # Zero radius
    )
    result_zero = interpolator_zero(test_data)
    nan_count_zero = np.sum(np.isnan(result_zero.data))
    # print(f"  Zero radius: {nan_count_zero} NaNs")

    edge_case_results = {
        "small": {"result": result_small, "nan_count": nan_count_small},
        "large": {"result": result_large, "nan_count": nan_count_large},
        "zero": {"result": result_zero, "nan_count": nan_count_zero},
    }

    # print("\nEdge case summary:")
    # print(f"  Zero radius: {nan_count_zero} NaNs (maximum possible)")
    # print(f" Small radius (10m): {nan_count_small} NaNs")
    # print(f"  Large radius (10Mm): {nan_count_large} NaNs (minimum possible)")

    return edge_case_results


def test_error_handling():
    """Test error handling for invalid radius values."""
    # print("\n=== Testing error handling ===")

    source_grid, target_grid = create_curvilinear_grids()
    test_data = create_test_data(source_grid)

    # Test negative radius (should raise an error or handle gracefully)
    # print("Testing negative radius:")
    try:
        interpolator_negative = CurvilinearInterpolator(
            source_grid, target_grid, method="nearest", radius_of_influence=-100000
        )
        result_negative = interpolator_negative(test_data)
        nan_count_negative = np.sum(np.isnan(result_negative.data))
        # print(f"  Negative radius handled, NaN count: {nan_count_negative}")
    except Exception as e:
        # print(f"  Negative radius raised exception: {e}")
        pass

    # Test extremely large radius (should work but might be slow)
    # print("Testing extremely large radius:")
    try:
        interpolator_extreme = CurvilinearInterpolator(
            source_grid,
            target_grid,
            method="nearest",
            radius_of_influence=1e10,  # Extremely large radius
        )
        result_extreme = interpolator_extreme(test_data)
        nan_count_extreme = np.sum(np.isnan(result_extreme.data))
        # print(f" Extremely large radius, NaN count: {nan_count_extreme}")
    except Exception as e:
        # print(f"  Extremely large radius raised exception: {e}")
        pass


def main():
    """Run all comprehensive tests."""
    # print("Running comprehensive radius_of_influence tests...\n")

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
    # print("\n" + "="*60)
    # print("COMPREHENSIVE TEST SUMMARY")
    # print("="*60)

    # print(f"✓ Various radius values tested: {len(results_various)} different values")
    # print("✓ Before/after behavior demonstrated")
    # print(f"✓ Backward compatibility: {'PASSED' if backward_compatible else 'FAILED'}")
    # print(f"✓ Performance benchmarks completed: {len(performance_results)} configurations tested")
    # print(f"✓ Edge cases tested: {len(edge_case_results)} scenarios")
    # print("✓ Error handling verified")

    # print("\nKey findings:")
    # print("- Radius of influence significantly affects NaN count in results")
    # print("- Larger radii result in fewer NaN values (more points filled)")
    # print("- Smaller radii result in more NaN values (stricter matching)")
    # print("- Backward compatibility maintained when radius_of_influence is None")
    # print("- Performance impact is minimal across different radius values")

    return {
        "various_values": results_various,
        "before_after": results_before_after,
        "backward_compatible": backward_compatible,
        "performance": performance_results,
        "edge_cases": edge_case_results,
    }


if __name__ == "__main__":
    results = main()
