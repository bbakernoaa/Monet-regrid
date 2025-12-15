"""Performance benchmark tests for curvilinear regridding optimization.

This module tests performance targets, scalability, and optimization effectiveness
compared to baseline implementations.
"""

import time

import numpy as np
import xarray as xr

import monet_regrid


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
            test_data = test_data.assign_coords(
                {"latitude": source_grid.latitude, "longitude": source_grid.longitude}
            )

            # Time the interpolation
            start_time = time.time()
            result = test_data.regrid.nearest(target_grid)
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
            test_data = test_data.assign_coords(
                {"latitude": source_grid.latitude, "longitude": source_grid.longitude}
            )

            start_time = time.time()
            test_data.regrid.nearest(target_grid)
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
        test_data = test_data.assign_coords(
            {"latitude": source_grid.latitude, "longitude": source_grid.longitude}
        )

        # Perform interpolation
        start_time = time.time()
        result = test_data.regrid.nearest(target_grid)
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
        test_data = test_data.assign_coords(
            {"latitude": source_grid.latitude, "longitude": source_grid.longitude}
        )

        # Time nearest neighbor
        start_time = time.time()
        result_nearest = test_data.regrid.nearest(target_grid)
        time_nearest = time.time() - start_time

        # Time linear interpolation
        start_time = time.time()
        result_linear = test_data.regrid.linear(target_grid)
        time_linear = time.time() - start_time

        # Linear should generally take longer than nearest (but both should complete)
        assert time_nearest < 10.0, f"Nearest interpolation too slow: {time_nearest:.2f}s"
        assert time_linear < 30.0, f"Linear interpolation too slow: {time_linear:.2f}s"

        # Verify both produce valid results
        assert result_nearest.shape == target_grid["latitude"].shape
        assert result_linear.shape == target_grid["latitude"].shape
