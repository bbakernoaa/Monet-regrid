"""Test data generation utilities for curvilinear regridding tests.

This module provides utilities for generating test data with various characteristics,
grid types, and challenging scenarios for comprehensive testing.
"""

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# All imports have been updated from xarray_regrid to monet_regrid.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import xarray as xr


class TestDataGenerator:
    """Utilities for generating test data for curvilinear regridding tests."""

    @staticmethod
    def create_curvilinear_grid(
        ny: int,
        nx: int,
        lat_range: tuple[float, float] = (-90, 90),
        lon_range: tuple[float, float] = (-180, 180),
        perturbation: float = 1.0,
        grid_type: str = "wave",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a curvilinear grid with specified characteristics.

        Args:
            ny: Number of grid points in y direction
            nx: Number of grid points in x direction
            lat_range: Latitude range (min, max)
            lon_range: Longitude range (min, max)
            perturbation: Amplitude of perturbation to make grid curvilinear
            grid_type: Type of perturbation ('wave', 'turbulent', 'shear')

        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        # Create base rectilinear grid
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        if grid_type == "wave":
            # Wave-like perturbation
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny) * np.cos(2 * np.pi * x_idx / nx)
            lon_perturb = perturbation * np.cos(2 * np.pi * y_idx / ny) * np.sin(2 * np.pi * x_idx / nx)

        elif grid_type == "turbulent":
            # More turbulent perturbation
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            lat_perturb = perturbation * (
                0.5 * np.sin(4 * np.pi * y_idx / ny) * np.cos(3 * np.pi * x_idx / nx)
                + 0.3 * np.cos(6 * np.pi * y_idx / ny) * np.sin(5 * np.pi * x_idx / nx)
                + 0.2 * np.sin(8 * np.pi * y_idx / ny) * np.cos(7 * np.pi * x_idx / nx)
            )
            lon_perturb = perturbation * (
                0.5 * np.cos(4 * np.pi * y_idx / ny) * np.sin(3 * np.pi * x_idx / nx)
                + 0.3 * np.sin(6 * np.pi * y_idx / ny) * np.cos(5 * np.pi * x_idx / nx)
                + 0.2 * np.cos(8 * np.pi * y_idx / ny) * np.sin(7 * np.pi * x_idx / nx)
            )

        elif grid_type == "shear":
            # Shear-like perturbation
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            lat_perturb = perturbation * 0.1 * y_idx / ny * x_idx / nx
            lon_perturb = perturbation * 0.1 * (ny - y_idx) / ny * (nx - x_idx) / nx

        else:
            raise ValueError(f"Unknown grid type: {grid_type}")

        return lat_2d + lat_perturb, lon_2d + lon_perturb

    @staticmethod
    def create_test_dataset(
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        data_type: str = "temperature",
        noise_level: float = 0.0,
        seed: int | None = None,
    ) -> xr.DataArray:
        """Create test data array with specified characteristics.

        Args:
            source_lat: Source latitude grid
            source_lon: Source longitude grid
            data_type: Type of data ('temperature', 'precipitation', 'pressure', 'wind')
            noise_level: Level of random noise to add (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            xarray DataArray with test data
        """
        if seed is not None:
            np.random.seed(seed)

        ny, nx = source_lat.shape

        if data_type == "temperature":
            # Temperature field with realistic gradient
            base_temp = 280.0 + 20.0 * np.sin(np.pi * source_lat / 180)  # Latitudinal gradient
            temp_gradient = (
                10.0 * (source_lon - source_lon.min()) / (source_lon.max() - source_lon.min())
            )  # Longitudinal gradient
            data_values = base_temp + temp_gradient

            # Add realistic spatial correlation
            if noise_level > 0:
                noise = TestDataGenerator._generate_spatial_noise(ny, nx, noise_level * 5.0, seed)
                data_values += noise

            attrs = {"units": "K", "long_name": "Temperature", "standard_name": "air_temperature"}

        elif data_type == "precipitation":
            # Precipitation field (positive values, skewed distribution)
            base_precip = 5.0 * np.exp(-(((source_lat - 0) / 30) ** 2))  # Maximum at equator
            precip_gradient = 2.0 * np.cos(np.pi * source_lon / 180)  # Longitudinal variation
            data_values = np.maximum(0, base_precip + precip_gradient)

            # Add multiplicative noise for precipitation
            if noise_level > 0:
                noise = np.random.lognormal(0, noise_level * 0.5, (ny, nx))
                data_values *= noise

            attrs = {"units": "kg m-2 s-1", "long_name": "Precipitation Rate", "standard_name": "precipitation_flux"}

        elif data_type == "pressure":
            # Pressure field with realistic variation
            base_pressure = 101325.0 - 100.0 * source_lat  # Decrease with latitude
            data_values = base_pressure

            if noise_level > 0:
                noise = TestDataGenerator._generate_spatial_noise(ny, nx, noise_level * 50.0, seed)
                data_values += noise

            attrs = {"units": "Pa", "long_name": "Air Pressure", "standard_name": "air_pressure"}

        elif data_type == "wind":
            # Wind field as vector components
            u_component = 10.0 * np.sin(2 * np.pi * source_lon / 180)  # Zonal wind
            v_component = 5.0 * np.cos(np.pi * source_lat / 90)  # Meridional wind

            if noise_level > 0:
                u_noise = TestDataGenerator._generate_spatial_noise(ny, nx, noise_level * 2.0, seed)
                v_noise = TestDataGenerator._generate_spatial_noise(
                    ny, nx, noise_level * 2.0, seed + 1 if seed else None
                )
                u_component += u_noise
                v_component += v_noise

            # Return as magnitude
            data_values = np.sqrt(u_component**2 + v_component**2)

            attrs = {"units": "m s-1", "long_name": "Wind Speed", "standard_name": "wind_speed"}

        else:
            # Generic smooth field
            data_values = np.sin(source_lat / 10) * np.cos(source_lon / 10)

            if noise_level > 0:
                noise = TestDataGenerator._generate_spatial_noise(ny, nx, noise_level, seed)
                data_values += noise

            attrs = {"units": "1", "long_name": "Test Field", "standard_name": "test_field"}

        dims = ["y", "x"]
        coords = {"y": range(ny), "x": range(nx)}

        return xr.DataArray(data_values, dims=dims, coords=coords, attrs=attrs)

    @staticmethod
    def _generate_spatial_noise(ny: int, nx: int, amplitude: float, seed: int | None = None) -> np.ndarray:
        """Generate spatially correlated noise."""
        if seed is not None:
            np.random.seed(seed)

        # Generate noise with spatial correlation
        noise = np.random.randn(ny, nx)

        # Apply simple spatial smoothing
        from scipy.ndimage import gaussian_filter

        smoothed_noise = gaussian_filter(noise, sigma=1.0)

        # Normalize and scale
        smoothed_noise = smoothed_noise / np.std(smoothed_noise) * amplitude

        return smoothed_noise

    @staticmethod
    def create_polar_grid(
        ny: int, nx: int, pole: str = "north", distance_from_pole: float = 10.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a grid near the poles.

        Args:
            ny: Number of grid points in y direction
            nx: Number of grid points in x direction
            pole: 'north' or 'south'
            distance_from_pole: Distance from pole in degrees

        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        if pole == "north":
            lat_min = 90.0 - distance_from_pole
            lat_max = 90.0
        else:  # south pole
            lat_min = -90.0
            lat_max = -90.0 + distance_from_pole

        lon_min, lon_max = -180, 180

        # Create base grid
        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        # Add small perturbation to make it curvilinear
        perturbation = 0.1
        y_idx, x_idx = np.ogrid[0:ny, 0:nx]
        lat_perturb = perturbation * np.sin(2 * np.pi * y_idx / ny)
        lon_perturb = perturbation * np.cos(2 * np.pi * x_idx / nx)

        return lat_2d + lat_perturb, lon_2d + lon_perturb

    @staticmethod
    def create_dateline_grid(ny: int, nx: int, crossing_longitude: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
        """Create a grid that crosses the International Date Line.

        Args:
            ny: Number of grid points in y direction
            nx: Number of grid points in x direction
            crossing_longitude: Longitude where grid crosses (typically 180)

        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        lat_min, lat_max = -30, 30
        lon_min = crossing_longitude - 30
        lon_max = crossing_longitude + 30

        # Handle date line crossing
        if lon_min < -180:
            lon_min += 360
        if lon_max > 180:
            lon_max -= 360

        lat_grid = np.linspace(lat_min, lat_max, ny)
        lon_grid = np.linspace(lon_min, lon_max, nx)
        lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

        # Add perturbation
        perturbation = 1.0
        y_idx, x_idx = np.ogrid[0:ny, 0:nx]
        lat_perturb = perturbation * np.sin(3 * np.pi * y_idx / ny)
        lon_perturb = perturbation * np.cos(3 * np.pi * x_idx / nx)

        return lat_2d + lat_perturb, lon_2d + lon_perturb

    @staticmethod
    def create_challenging_grid(ny: int, nx: int, challenge_type: str = "degenerate") -> tuple[np.ndarray, np.ndarray]:
        """Create grids that challenge the interpolation algorithms.

        Args:
            ny: Number of grid points in y direction
            nx: Number of grid points in x direction
            challenge_type: Type of challenge ('degenerate', 'high_aspect', 'singular')

        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        if challenge_type == "degenerate":
            # Nearly collinear points
            lat_base = np.linspace(0, 1, ny)
            lon_base = np.linspace(0, 1, nx)
            lat_2d, lon_2d = np.meshgrid(lat_base, lon_base, indexing="ij")

            # Add tiny perturbations
            eps = 1e-8
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            lat_2d += eps * np.sin(y_idx)
            lon_2d += eps * np.cos(x_idx)

        elif challenge_type == "high_aspect":
            # High aspect ratio grid
            lat_base = np.linspace(0, 0.1, ny)  # Very small latitude range
            lon_base = np.linspace(0, 180, nx)  # Large longitude range
            lat_2d, lon_2d = np.meshgrid(lat_base, lon_base, indexing="ij")

            # Add perturbation
            perturbation = 0.01
            y_idx, x_idx = np.ogrid[0:ny, 0:nx]
            lat_2d += perturbation * np.sin(10 * y_idx / ny)
            lon_2d += perturbation * np.cos(10 * x_idx / nx)

        elif challenge_type == "singular":
            # Grid with identical or very close points
            lat_base = np.ones(ny) * 45.0  # All same latitude
            lon_base = np.ones(nx) * 0.0  # All same longitude

            lat_2d, lon_2d = np.meshgrid(lat_base, lon_base, indexing="ij")

            # Add tiny differences
            eps = 1e-10
            for i in range(ny):
                for j in range(nx):
                    lat_2d[i, j] += i * eps
                    lon_2d[i, j] += j * eps
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")

        return lat_2d, lon_2d

    @staticmethod
    def create_multidimensional_data(
        base_data: xr.DataArray, time_steps: int = 5, levels: int = 3, add_temporal_correlation: bool = True
    ) -> xr.DataArray:
        """Create multidimensional test data with time and level dimensions.

        Args:
            base_data: 2D base data array
            time_steps: Number of time steps
            levels: Number of vertical levels
            add_temporal_correlation: Whether to add temporal correlation

        Returns:
            4D xarray DataArray
        """
        ny, nx = base_data.shape[-2:]

        # Create base 4D array
        data_4d = np.tile(base_data.values[np.newaxis, np.newaxis, :, :], (levels, time_steps, 1, 1))

        if add_temporal_correlation:
            # Add realistic temporal evolution
            for t in range(time_steps):
                for lev in range(levels):
                    # Add time and level dependent variations
                    time_factor = 1 + 0.1 * np.sin(2 * np.pi * t / time_steps)
                    level_factor = 1 + 0.2 * lev / levels
                    noise_factor = 0.05 * np.random.randn(ny, nx)

                    data_4d[lev, t] *= time_factor * level_factor * (1 + noise_factor)

        # Create coordinates
        dims = ["level", "time", "y", "x"]
        coords = {
            "level": range(levels),
            "time": range(time_steps),
            "y": base_data.coords["y"],
            "x": base_data.coords["x"],
        }

        return xr.DataArray(data_4d, dims=dims, coords=coords, attrs=base_data.attrs)

    @staticmethod
    def create_dataset_with_nans(
        base_data: xr.DataArray, nan_percentage: float = 10.0, nan_pattern: str = "random", seed: int | None = None
    ) -> xr.DataArray:
        """Create data with NaN values in specified patterns.

        Args:
            base_data: Base data array
            nan_percentage: Percentage of values to set as NaN
            nan_pattern: Pattern of NaNs ('random', 'clustered', 'striped', 'edge')
            seed: Random seed for reproducibility

        Returns:
            DataArray with NaN values
        """
        if seed is not None:
            np.random.seed(seed)

        data_values = base_data.values.copy()
        ny, nx = data_values.shape

        total_points = ny * nx
        nan_count = int(total_points * nan_percentage / 100)

        if nan_pattern == "random":
            # Random NaN placement
            indices = np.random.choice(total_points, nan_count, replace=False)
            flat_indices = np.unravel_index(indices, (ny, nx))
            data_values[flat_indices] = np.nan

        elif nan_pattern == "clustered":
            # Clustered NaN regions
            cluster_size = max(1, int(np.sqrt(nan_count)))
            clusters = nan_count // (cluster_size**2)

            for _ in range(clusters):
                center_y = np.random.randint(0, ny)
                center_x = np.random.randint(0, nx)

                for dy in range(-cluster_size // 2, cluster_size // 2 + 1):
                    for dx in range(-cluster_size // 2, cluster_size // 2 + 1):
                        y_idx = max(0, min(ny - 1, center_y + dy))
                        x_idx = max(0, min(nx - 1, center_x + dx))
                        data_values[y_idx, x_idx] = np.nan

        elif nan_pattern == "striped":
            # NaN in stripes
            num_stripes = nan_count // 10
            if num_stripes == 0:
                num_stripes = 1
            stripe_width = max(1, ny // num_stripes)
            for i in range(0, ny, stripe_width):
                data_values[i : min(i + stripe_width // 3, ny), :] = np.nan

        elif nan_pattern == "edge":
            # NaN at edges
            edge_width = max(1, int(ny * nan_percentage / 200))
            data_values[:edge_width, :] = np.nan
            data_values[-edge_width:, :] = np.nan
            data_values[:, :edge_width] = np.nan
            data_values[:, -edge_width:] = np.nan
        else:
            raise ValueError(f"Unknown NaN pattern: {nan_pattern}")

        return xr.DataArray(data_values, dims=base_data.dims, coords=base_data.coords, attrs=base_data.attrs)

    @staticmethod
    def create_extreme_coordinates(coordinate_type: str = "bounds") -> tuple[np.ndarray, np.ndarray]:
        """Create grids with extreme coordinate values.

        Args:
            coordinate_type: Type of extreme coordinates ('bounds', 'pole', 'dateline')

        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        if coordinate_type == "bounds":
            # Coordinates at extreme bounds
            lat_vals = np.array([[90.0, -90.0], [89.999, -89.999]])
            lon_vals = np.array([[180.0, -180.0], [179.999, -179.999]])

        elif coordinate_type == "pole":
            # Coordinates very close to poles
            lat_vals = np.array([[89.9999, 89.9999], [-89.9999, -89.9999]])
            lon_vals = np.array([[0.0, 180.0], [0.0, 180.0]])

        elif coordinate_type == "dateline":
            # Coordinates crossing dateline
            lat_vals = np.array([[0.0, 10.0], [0.0, 10.0]])
            lon_vals = np.array([[179.999, 179.999], [-179.999, -179.999]])

        else:
            raise ValueError(f"Unknown coordinate type: {coordinate_type}")

        return lat_vals, lon_vals


class TestGridTypes:
    """Test different types of grid configurations."""

    def test_regular_curvilinear_grids(self):
        """Test generation of regular curvilinear grids."""
        # Test different grid types
        for grid_type in ["wave", "turbulent", "shear"]:
            lat, lon = TestDataGenerator.create_curvilinear_grid(ny=10, nx=12, perturbation=2.0, grid_type=grid_type)

            assert lat.shape == (10, 12)
            assert lon.shape == (10, 12)
            assert np.all(np.isfinite(lat))
            assert np.all(np.isfinite(lon))

    def test_polar_grids(self):
        """Test generation of polar grids."""
        for pole in ["north", "south"]:
            lat, lon = TestDataGenerator.create_polar_grid(ny=8, nx=10, pole=pole, distance_from_pole=15.0)

            assert lat.shape == (8, 10)
            assert lon.shape == (8, 10)

            if pole == "north":
                assert np.all(lat >= 75.0)  # Within 15 degrees of North Pole
                assert np.all(lat <= 90.0)
            else:
                assert np.all(lat >= -90.0)
                assert np.all(lat <= -75.0)  # Within 15 degrees of South Pole

    def test_dateline_grids(self):
        """Test generation of grids crossing the date line."""
        lat, lon = TestDataGenerator.create_dateline_grid(ny=6, nx=8)

        assert lat.shape == (6, 8)
        assert lon.shape == (6, 8)
        # Should have coordinates on both sides of dateline
        assert np.any(lon > 0) and np.any(lon < 0)

    def test_challenging_grids(self):
        """Test generation of challenging grid configurations."""
        for challenge_type in ["degenerate", "high_aspect", "singular"]:
            lat, lon = TestDataGenerator.create_challenging_grid(ny=5, nx=5, challenge_type=challenge_type)

            assert lat.shape == (5, 5)
            assert lon.shape == (5, 5)
            assert np.all(np.isfinite(lat) | np.isnan(lat))
            assert np.all(np.isfinite(lon) | np.isnan(lon))


class TestDataCharacteristics:
    """Test different data characteristics and patterns."""

    def test_different_data_types(self):
        """Test generation of different data types."""
        lat, lon = TestDataGenerator.create_curvilinear_grid(5, 6)

        for data_type in ["temperature", "precipitation", "pressure", "wind"]:
            test_data = TestDataGenerator.create_test_dataset(lat, lon, data_type=data_type, noise_level=0.1, seed=42)

            assert test_data.shape == (5, 6)
            assert "units" in test_data.attrs
            assert "long_name" in test_data.attrs

            # Check data type specific characteristics
            if data_type == "precipitation":
                assert np.all(test_data >= 0)  # Precipitation should be non-negative

    def test_noise_levels(self):
        """Test generation with different noise levels."""
        lat, lon = TestDataGenerator.create_curvilinear_grid(5, 6)

        base_data = TestDataGenerator.create_test_dataset(lat, lon, data_type="temperature", noise_level=0.0)

        for noise_level in [0.0, 0.1, 0.5]:
            noisy_data = TestDataGenerator.create_test_dataset(
                lat, lon, data_type="temperature", noise_level=noise_level, seed=42
            )

            assert noisy_data.shape == (5, 6)

            if noise_level == 0.0:
                # Should be identical to base data
                np.testing.assert_array_equal(base_data.values, noisy_data.values)

    def test_multidimensional_data(self):
        """Test generation of multidimensional data."""
        base_lat, base_lon = TestDataGenerator.create_curvilinear_grid(4, 5)
        base_data = TestDataGenerator.create_test_dataset(base_lat, base_lon, data_type="temperature")

        # Test with different dimensions
        test_data = TestDataGenerator.create_multidimensional_data(
            base_data, time_steps=3, levels=2, add_temporal_correlation=True
        )

        assert test_data.shape == (2, 3, 4, 5)  # (level, time, y, x)
        assert "level" in test_data.dims
        assert "time" in test_data.dims
        assert test_data.attrs == base_data.attrs

    def test_nan_patterns(self):
        """Test generation of data with different NaN patterns."""
        lat, lon = TestDataGenerator.create_curvilinear_grid(6, 8)
        base_data = TestDataGenerator.create_test_dataset(lat, lon, data_type="temperature")

        for nan_pattern in ["random", "clustered", "striped", "edge"]:
            nan_data = TestDataGenerator.create_dataset_with_nans(
                base_data, nan_percentage=20.0, nan_pattern=nan_pattern, seed=42
            )

            assert nan_data.shape == (6, 8)

            # Check that NaNs were introduced
            nan_count = np.sum(np.isnan(nan_data))
            assert nan_count > 0
            assert nan_count <= 0.55 * 6 * 8  # Should be around 20% NaN, allow some margin for edge patterns

    def test_extreme_coordinates(self):
        """Test generation of extreme coordinate configurations."""
        for coord_type in ["bounds", "pole", "dateline"]:
            lat, lon = TestDataGenerator.create_extreme_coordinates(coordinate_type=coord_type)

            assert lat.shape == (2, 2)
            assert lon.shape == (2, 2)

            # Check coordinate bounds
            assert np.all(lat >= -90) and np.all(lat <= 90)
            assert np.all(lon >= -180) and np.all(lon <= 180)


if __name__ == "__main__":
    # Run data generation tests
    grid_test = TestGridTypes()
    grid_test.test_regular_curvilinear_grids()
    grid_test.test_polar_grids()
    grid_test.test_dateline_grids()
    grid_test.test_challenging_grids()

    data_test = TestDataCharacteristics()
    data_test.test_different_data_types()
    data_test.test_noise_levels()
    data_test.test_multidimensional_data()
    data_test.test_nan_patterns()
    data_test.test_extreme_coordinates()

    print("All data generation tests passed!")
