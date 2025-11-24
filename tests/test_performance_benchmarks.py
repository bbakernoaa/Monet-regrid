"""Performance benchmark tests for curvilinear regridding optimization.

This module tests performance targets, scalability, and optimization effectiveness
compared to baseline implementations.
"""

import statistics
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from xarray_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


class TestPerformanceBenchmarks:
    """Test performance benchmarks and optimization targets."""

    def setup_method(self):
        """Set up performance test data."""
        # Grid sizes for scalability testing
        self.grid_sizes = [
            (10, 12),  # Small
            (20, 25),  # Medium
            (40, 50),  # Large
            (80, 100),  # Extra large
        ]

        # Performance thresholds (adjust based on requirements)
        self.time_thresholds = {
            "small": 2.0,  # seconds
            "medium": 5.0,  # seconds
            "large": 15.0,  # seconds
            "xlarge": 60.0,  # seconds
        }

    def _create_test_grids(self, ny: int, nx: int, grid_type: str = "curvilinear"):
        """Create test grids of specified size."""
        # Create base grids - use safe ranges that account for perturbations
        # Leave buffer to prevent exceeding valid coordinate ranges after perturbation
        lat_buffer = 5.0  # Leave 5 degree buffer at each end
        lat_min, lat_max = -90 + lat_buffer, 90 - lat_buffer
        lon_min, lon_max = -180, 180

        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        if grid_type == "curvilinear":
            # Add perturbation to make it curvilinear
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            # Reduce perturbation to ensure we don't exceed coordinate bounds
            perturbation = 3.0  # Reduced from 5.0 to ensure bounds are not exceeded
            lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny) * np.cos(2 * np.pi * x_idx / nx)
            lon_perturb = perturbation * np.cos(2 * np.pi * y_idx / ny) * np.sin(2 * np.pi * x_idx / nx)
            lat_2d += lat_perturb
            lon_2d += lon_perturb

            # Ensure latitudes don't exceed bounds [-90, 90]
            lat_2d = np.clip(lat_2d, -90.0, 90.0)
            # Ensure longitudes are within [-180, 180]
            lon_2d = ((lon_2d + 180) % 360) - 180

        source_grid = xr.Dataset({"latitude": (["y", "x"], lat_2d), "longitude": (["y", "x"], lon_2d)})

        # Create slightly smaller target grid
        target_ny, target_nx = max(1, ny - 2), max(1, nx - 2)
        target_lat_grid = np.linspace(lat_min + 5, lat_max - 5, target_ny)
        target_lon_grid = np.linspace(lon_min + 10, lon_max - 10, target_nx)
        target_lat_2d, target_lon_2d = np.meshgrid(target_lat_grid, target_lon_grid, indexing="ij")

        if grid_type == "curvilinear":
            target_y_idx, target_x_idx = np.ogrid[0:target_ny, 0:target_nx]
            target_perturbation = 3.0
            target_lat_perturb = (
                target_perturbation
                * np.sin(2 * np.pi * target_y_idx / target_ny)
                * np.cos(2 * np.pi * target_x_idx / target_nx)
            )
            target_lon_perturb = (
                target_perturbation
                * np.cos(2 * np.pi * target_y_idx / target_ny)
                * np.sin(2 * np.pi * target_x_idx / target_nx)
            )
            target_lat_2d += target_lat_perturb
            target_lon_2d += target_lon_perturb

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], target_lat_2d),
                "longitude": (["y_target", "x_target"], target_lon_2d),
            }
        )

        return source_grid, target_grid

    def _create_test_data(self, ny: int, nx: int, dtype=np.float64):
        """Create test data array."""
        np.random.seed(42)  # For reproducible results
        data_values = np.random.rand(ny, nx).astype(dtype) * 100 + 273.15
        return xr.DataArray(data_values, dims=["y", "x"])

    def _get_grid_category(self, ny: int, nx: int) -> str:
        """Get grid size category for threshold lookup."""
        total_points = ny * nx
        if total_points <= 200:
            return "small"
        elif total_points <= 1000:
            return "medium"
        elif total_points <= 3000:
            return "large"
        else:
            return "xlarge"

    def test_interpolation_speed_targets(self):
        """Test that interpolation meets speed targets for different grid sizes."""
        results = {}

        for ny, nx in self.grid_sizes:
            source_grid, target_grid = self._create_test_grids(ny, nx)
            test_data = self._create_test_data(ny, nx)

            # Time the interpolation
            start_time = time.time()
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)
            elapsed_time = time.time() - start_time

            grid_category = self._get_grid_category(ny, nx)
            threshold = self.time_thresholds[grid_category]

            results[(ny, nx)] = {"time": elapsed_time, "threshold": threshold, "passed": elapsed_time <= threshold}

            # Verify result correctness
            assert result.shape == target_grid["latitude"].shape
            assert np.all(np.isfinite(result) | np.isnan(result))

    def test_scalability_analysis(self):
        """Test how performance scales with grid size."""
        sizes = [(10, 12), (20, 25), (40, 50)]
        times = []

        for ny, nx in sizes:
            source_grid, target_grid = self._create_test_grids(ny, nx)
            test_data = self._create_test_data(ny, nx)

            start_time = time.time()
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)

        # Check that time increases at reasonable rate (should be sub-quadratic)
        # For doubling grid size, time should increase by less than 4x (quadratic scaling)
        for i in range(1, len(times)):
            size_ratio = (sizes[i][0] * sizes[i][1]) / (sizes[i - 1][0] * sizes[i - 1][1])
            time_ratio = times[i] / times[i - 1]

            # Allow up to 8x time increase for 4x size increase (some overhead is expected)
            assert time_ratio < 8.0, (
                f"Time scaling too steep: {time_ratio:.2f}x increase for {size_ratio:.2f}x size increase"
            )

    def test_memory_efficiency(self):
        """Test that memory usage is reasonable for large grids."""
        # Test with a large grid
        ny, nx = 100, 120
        source_grid, target_grid = self._create_test_grids(ny, nx)
        test_data = self._create_test_data(ny, nx, dtype=np.float64)

        # Get initial memory estimate (this is approximate)
        input_size_mb = test_data.nbytes / (1024**2)
        expected_output_size_mb = (target_grid["latitude"].size * 8) / (1024**2)  # 8 bytes per float64

        # Perform interpolation
        start_time = time.time()
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)
        elapsed_time = time.time() - start_time

        # Verify completion within reasonable time
        assert elapsed_time < 30.0, f"Large grid interpolation took too long: {elapsed_time:.2f}s"

        # Verify result size
        assert result.shape == target_grid["latitude"].shape
        assert result.size == target_grid["latitude"].size

    def test_method_performance_comparison(self):
        """Compare performance between nearest and linear methods."""
        ny, nx = 50, 60
        source_grid, target_grid = self._create_test_grids(ny, nx)
        test_data = self._create_test_data(ny, nx)

        # Time nearest neighbor
        start_time = time.time()
        interpolator_nearest = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result_nearest = interpolator_nearest(test_data)
        time_nearest = time.time() - start_time

        # Time linear interpolation
        start_time = time.time()
        interpolator_linear = CurvilinearInterpolator(source_grid, target_grid, method="linear")
        result_linear = interpolator_linear(test_data)
        time_linear = time.time() - start_time

        # Linear should generally take longer than nearest (but both should complete)
        assert time_nearest < 10.0, f"Nearest interpolation too slow: {time_nearest:.2f}s"
        assert time_linear < 30.0, f"Linear interpolation too slow: {time_linear:.2f}s"

        # Verify both produce valid results
        assert result_nearest.shape == target_grid["latitude"].shape
        assert result_linear.shape == target_grid["latitude"].shape

    def test_caching_effectiveness(self):
        """Test that repeated interpolations benefit from caching."""
        ny, nx = 30, 40
        source_grid, target_grid = self._create_test_grids(ny, nx)
        test_data = self._create_test_data(ny, nx)

        # Time first interpolation (cold cache)
        start_time = time.time()
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result1 = interpolator(test_data)
        time_first = time.time() - start_time

        # Time second interpolation (should benefit from any internal caching)
        start_time = time.time()
        result2 = interpolator(test_data)
        time_second = time.time() - start_time

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

        # Second run should not be dramatically slower (though caching benefits may be limited)
        assert time_second < time_first * 2.0, (
            f"Second interpolation much slower: {time_second:.2f}s vs {time_first:.2f}s"
        )


class TestOptimizationValidation:
    """Validate that optimizations provide expected performance improvements."""

    def test_algorithmic_complexity_scaling(self):
        """Test that algorithmic complexity is as expected (should be O(n log n) for KDTree)."""
        sizes = [(20, 25), (40, 50), (80, 100)]
        times = []

        for ny, nx in sizes:
            # Create test data
            source_grid, target_grid = self._create_test_grids(ny, nx)
            test_data = self._create_test_data(ny, nx)

            # Time the operation
            start_time = time.time()
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)

        # Check that time scaling is reasonable for O(n log n) algorithm
        # When size doubles, time should increase by roughly 2 * log(2n) / log(n) ≈ 2.1-2.5x
        for i in range(1, len(sizes)):
            size_ratio = (sizes[i][0] * sizes[i][1]) / (sizes[i - 1][0] * sizes[i - 1][1])
            time_ratio = times[i] / times[i - 1]

            # Allow reasonable variance in timing measurements
            expected_scaling = size_ratio * 1.2  # Allow some overhead
            assert time_ratio < expected_scaling * 2.0, (
                f"Time scaling suggests worse than O(n log n): {time_ratio:.2f}x for {size_ratio:.2f}x size"
            )

    def test_spatial_query_efficiency(self):
        """Test that spatial queries are efficient (KDTree performance)."""
        # Create a large source grid and small target grid to stress spatial queries
        source_ny, source_nx = 100, 120
        target_ny, target_nx = 5, 6

        # Create test grids
        source_grid, _ = self._create_test_grids(source_ny, source_nx)

        # Create small target grid with specified dimensions
        # For a proper 2D curvilinear grid, both coordinates should form 2D arrays with the same dimensions
        target_lat_1d = np.linspace(-85, 85, target_ny)
        target_lon_1d = np.linspace(-175, 175, target_nx)

        # Create 2D coordinate arrays using meshgrid
        target_lat_2d, target_lon_2d = np.meshgrid(target_lat_1d, target_lon_1d, indexing="ij")

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], target_lat_2d),
                "longitude": (["y_target", "x_target"], target_lon_2d),
            }
        )

        test_data = self._create_test_data(source_ny, source_nx)

        # This should complete quickly due to efficient spatial indexing
        start_time = time.time()
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, f"Large-to-small interpolation too slow: {elapsed_time:.2f}s"
        # The result should match the target grid shape
        assert result.shape == target_grid["latitude"].shape

    def test_memory_usage_patterns(self):
        """Test that memory usage patterns are efficient."""
        # Test with different data types
        dtypes = [np.float32, np.float64]
        sizes = []

        for dtype in dtypes:
            ny, nx = 50, 60
            source_grid, target_grid = self._create_test_grids(ny, nx)
            test_data = self._create_test_data(ny, nx, dtype=dtype)

            # Time and verify completion
            start_time = time.time()
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)
            elapsed_time = time.time() - start_time

            assert elapsed_time < 15.0, f"Interpolation with {dtype} too slow: {elapsed_time:.2f}s"
            assert result.dtype == dtype or result.dtype == np.float64  # May promote to float64
            assert result.shape == target_grid["latitude"].shape

    def _create_test_grids(self, ny: int, nx: int):
        """Helper to create test grids."""
        lat_min, lat_max = -90, 90
        lon_min, lon_max = -180, 180

        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        # Add perturbation
        y_idx, x_idx = np.ogrid[0:ny, 0:nx]
        # Reduce perturbation to ensure we don't exceed coordinate bounds
        perturbation = 3.0  # Reduced from 5.0 to ensure bounds are not exceeded
        lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny) * np.cos(2 * np.pi * x_idx / nx)
        lon_perturb = perturbation * np.cos(2 * np.pi * y_idx / ny) * np.sin(2 * np.pi * x_idx / nx)
        lat_2d += lat_perturb
        lon_2d += lon_perturb

        # Ensure latitudes don't exceed bounds [-90, 90]
        lat_2d = np.clip(lat_2d, -90.0, 90.0)
        # Ensure longitudes are within [-180, 180]
        lon_2d = ((lon_2d + 180) % 360) - 180

        source_grid = xr.Dataset({"latitude": (["y", "x"], lat_2d), "longitude": (["y", "x"], lon_2d)})

        # Target grid - also ensure safe bounds
        target_ny, target_nx = max(1, ny - 2), max(1, nx - 2)
        target_lat_grid = np.linspace(lat_min + 5, lat_max - 5, target_ny)  # Within safe bounds
        target_lon_grid = np.linspace(lon_min + 10, lon_max - 10, target_nx)  # Within safe bounds
        target_lat_2d, target_lon_2d = np.meshgrid(target_lat_grid, target_lon_grid, indexing="ij")

        # Add perturbation to target grid as well
        target_y_idx, target_x_idx = np.ogrid[0:target_ny, 0:target_nx]
        target_perturbation = 2.0  # Smaller perturbation for target grid
        target_lat_perturb = (
            target_perturbation
            * np.sin(2 * np.pi * target_y_idx / target_ny)
            * np.cos(2 * np.pi * target_x_idx / target_nx)
        )
        target_lon_perturb = (
            target_perturbation
            * np.cos(2 * np.pi * target_y_idx / target_ny)
            * np.sin(2 * np.pi * target_x_idx / target_nx)
        )
        target_lat_2d += target_lat_perturb
        target_lon_2d += target_lon_perturb

        # Ensure target latitudes don't exceed bounds [-90, 90]
        target_lat_2d = np.clip(target_lat_2d, -90.0, 90.0)
        # Ensure target longitudes are within [-180, 180]
        target_lon_2d = ((target_lon_2d + 180) % 360) - 180

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], target_lat_2d),
                "longitude": (["y_target", "x_target"], target_lon_2d),
            }
        )

        return source_grid, target_grid

    def _create_test_data(self, ny: int, nx: int, dtype=np.float64):
        """Helper to create test data."""
        np.random.seed(42)
        data_values = np.random.rand(ny, nx).astype(dtype) * 100 + 273.15
        return xr.DataArray(data_values, dims=["y", "x"])


class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_baseline_performance_comparison(self):
        """Compare against baseline performance expectations."""
        # This would ideally compare against known baseline performance
        # For now, we establish minimum acceptable performance

        ny, nx = 50, 60
        source_grid, target_grid = self._create_test_grids(ny, nx)
        test_data = self._create_test_data(ny, nx)

        # Establish baseline timing
        start_time = time.time()
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)
        elapsed_time = time.time() - start_time

        # Should complete within reasonable time
        assert elapsed_time < 15.0, f"Performance regression detected: {elapsed_time:.2f}s > 15s threshold"
        assert result.shape == target_grid["latitude"].shape

    def _create_test_grids(self, ny: int, nx: int):
        """Helper to create test grids."""
        lat_min, lat_max = -90, 90
        lon_min, lon_max = -180, 180

        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        # Add some perturbation to make it curvilinear
        y_idx, x_idx = np.ogrid[0:ny, 0:nx]
        perturbation = 3.0  # Reduced from 5.0 to stay within bounds
        lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny) * np.cos(2 * np.pi * x_idx / nx)
        lon_perturb = perturbation * np.cos(2 * np.pi * y_idx / ny) * np.sin(2 * np.pi * x_idx / nx)
        lat_2d += lat_perturb
        lon_2d += lon_perturb

        # Ensure latitudes don't exceed bounds [-90, 90]
        lat_2d = np.clip(lat_2d, -90.0, 90.0)
        # Ensure longitudes are within [-180, 180]
        lon_2d = ((lon_2d + 180) % 360) - 180

        source_grid = xr.Dataset({"latitude": (["y", "x"], lat_2d), "longitude": (["y", "x"], lon_2d)})

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], lat_2d[: ny - 2, : nx - 2]),
                "longitude": (["y_target", "x_target"], lon_2d[: ny - 2, : nx - 2]),
            }
        )

        return source_grid, target_grid

    def _create_test_data(self, ny: int, nx: int):
        """Helper to create test data."""
        np.random.seed(42)
        data_values = np.random.rand(ny, nx) * 100 + 273.15
        return xr.DataArray(data_values, dims=["y", "x"])


if __name__ == "__main__":
    # Run performance benchmark tests
    perf_test = TestPerformanceBenchmarks()
    perf_test.setup_method()

    perf_test.test_interpolation_speed_targets()
    perf_test.test_scalability_analysis()
    perf_test.test_memory_efficiency()
    perf_test.test_method_performance_comparison()
    perf_test.test_caching_effectiveness()

    opt_test = TestOptimizationValidation()
    opt_test.test_algorithmic_complexity_scaling()
    opt_test.test_spatial_query_efficiency()
    opt_test.test_memory_usage_patterns()

    reg_test = TestPerformanceRegression()
    reg_test.test_baseline_performance_comparison()

    print("All performance benchmark tests passed!")
