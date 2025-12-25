"""Integration tests for curvilinear regridding end-to-end workflows.

This module tests complete interpolation workflows, performance comparisons,
and component interactions in the curvilinear regridding system.
"""

import time

import numpy as np
import xarray as xr
import monet_regrid  # noqa: F401


class TestEndToEndWorkflows:
    """Test complete end-to-end interpolation workflows."""

    def setup_method(self):
        """Set up test data for integration tests."""
        # Create realistic curvilinear grids
        self.source_lat, self.source_lon = self._create_curvilinear_grid(10, 12, 30, 50, -100, -80)
        self.target_lat, self.target_lon = self._create_curvilinear_grid(8, 10, 32, 48, -98, -82, perturbation=0.3)

        self.source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.source_lat), "longitude": (["y", "x"], self.source_lon)}
        )

        self.target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.target_lat),
                "longitude": (["y_target", "x_target"], self.target_lon),
            }
        )

    def _create_curvilinear_grid(self, ny, nx, lat_min, lat_max, lon_min, lon_max, perturbation=0.1):
        """Create a curvilinear grid with some perturbation."""
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        # Add some perturbation to make it truly curvilinear
        y_idx, x_idx = np.ogrid[0:ny, 0:nx]
        lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny) * np.cos(2 * np.pi * x_idx / nx)
        lon_perturb = perturbation * np.cos(2 * np.pi * y_idx / ny) * np.sin(2 * np.pi * x_idx / nx)

        lat_result = lat_2d + lat_perturb
        lon_result = lon_2d + lon_perturb

        # Ensure latitude values are within valid range [-90, 90]
        lat_result = np.clip(lat_result, -90.0, 90.0)

        # Ensure longitude values are within valid range [-180, 180]
        lon_result = ((lon_result + 180) % 360) - 180

        return lat_result, lon_result

    def test_curvilinear_to_curvilinear_regridding(self):
        """Test complete curvilinear-to-curvilinear regridding workflow."""
        # Create test data
        data_values = np.random.rand(10, 12) * 100 + 273.15  # Temperature in Kelvin
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
            attrs={"units": "K", "long_name": "Temperature"},
        )

        # Test nearest neighbor interpolation
        result = test_data.regrid.nearest(self.target_grid)

        # Verify result properties
        assert result.shape == self.target_lat.shape
        assert result.shape == self.target_grid["latitude"].shape
        assert result.attrs["units"] == test_data.attrs["units"]
        assert result.attrs["long_name"] == test_data.attrs["long_name"]

        # Verify coordinate values are reasonable
        assert np.all(np.isfinite(result.values))
        assert np.all(result.values >= 273.15 - 100)
        assert np.all(result.values <= 273.15 + 200)

    def test_curvilinear_to_rectilinear_regridding(self):
        """Test curvilinear-to-rectilinear regridding workflow."""
        # Create rectilinear target grid
        target_lat_1d = np.linspace(32, 48, 8)
        target_lon_1d = np.linspace(-98, -82, 10)

        rectilinear_target = xr.Dataset(
            {"latitude": (["y_target"], target_lat_1d), "longitude": (["x_target"], target_lon_1d)}
        )

        # Create test data for this specific test
        data_values = np.random.rand(10, 12) * 10 + 100
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )

        # Test interpolation
        result = test_data.regrid.linear(rectilinear_target)

        # Verify result shape matches rectilinear target
        assert result.shape == (8, 10)

    def test_dataset_interpolation_workflow(self):
        """Test interpolation of multi-variable datasets."""
        # Create multi-variable dataset
        temp_values = np.random.rand(10, 12) * 50 + 273.15
        precip_values = np.random.rand(10, 12) * 10
        pressure_values = np.random.rand(10, 12) * 1000 + 100000

        test_dataset = xr.Dataset(
            {
                "temperature": (["y", "x"], temp_values),
                "precipitation": (["y", "x"], precip_values),
                "pressure": (["y", "x"], pressure_values),
                "static_field": ("time", [1, 2, 3]),
            },
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )

        # Test interpolation
        result = test_dataset.regrid.nearest(self.target_grid)

        # Verify all variables are present
        assert "temperature" in result
        assert "precipitation" in result
        assert "pressure" in result
        assert "static_field" in result

        # Verify interpolated variables have correct shape
        assert result["temperature"].shape == self.target_grid["latitude"].shape
        assert result["precipitation"].shape == self.target_grid["latitude"].shape
        assert result["pressure"].shape == self.target_grid["latitude"].shape

        # Verify static field is preserved
        assert result["static_field"].shape == (3,)
        np.testing.assert_array_equal(result["static_field"], [1, 2, 3])

    def test_multidimensional_data_interpolation(self):
        """Test interpolation with additional dimensions (time, level)."""
        # Create data with time and level dimensions
        time_dim = 5
        level_dim = 3
        data_values = np.random.rand(time_dim, level_dim, 10, 12)

        test_data = xr.DataArray(
            data_values,
            dims=["time", "level", "y", "x"],
            coords={
                "time": np.arange(time_dim),
                "level": np.arange(level_dim),
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )

        # Test interpolation
        result = test_data.regrid.nearest(self.target_grid)

        # Verify result dimensions
        expected_shape = (time_dim, level_dim, *self.target_grid["latitude"].shape)
        assert result.shape == expected_shape
        assert "time" in result.dims
        assert "level" in result.dims
