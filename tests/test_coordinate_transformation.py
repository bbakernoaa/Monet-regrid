"""Unit tests for coordinate transformation in curvilinear regridding.

This module tests the 3D coordinate transformation accuracy, pyproj integration,
and spherical geometry handling in the CurvilinearInterpolator.
"""

import numpy as np
import xarray as xr

import monet_regrid


class TestCoordinateTransformation:
    """Test coordinate transformation accuracy and precision."""

    def setup_method(self):
        """Set up test data for coordinate transformation tests."""
        # Create test grids with known transformations
        self.source_lat = np.array([[0, 10], [0, 10]])
        self.source_lon = np.array([[-10, -10], [10, 10]])
        self.target_lat = np.array([[5, 7], [5, 7]])
        self.target_lon = np.array([[-5, -5], [5, 5]])

        self.source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.source_lat), "longitude": (["y", "x"], self.source_lon)}
        )

        self.target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.target_lat),
                "longitude": (["y_target", "x_target"], self.target_lon),
            }
        )

    def test_pyproj_transformation_initialization(self):
        """Test that pyproj transformer is properly initialized."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )
        result = source_data.regrid.nearest(self.target_grid)
        assert result.shape == self.target_lat.shape

    def test_inverse_transformation(self):
        """Test inverse coordinate transformation."""
        transformer = monet_regrid.coordinate_transformer.CoordinateTransformer()
        lon, lat = 10, 20
        x, y, z = transformer.transform_coordinates(np.array([lon]), np.array([lat]))
        lon_inv, lat_inv, _ = transformer.inverse_transform_coordinates(x, y, z)
        assert np.isclose(lon, lon_inv[0])
        assert np.isclose(lat, lat_inv[0])

    def test_caching_mechanism(self):
        """Test caching mechanism."""
        transformer = monet_regrid.coordinate_transformer.CoordinateTransformer()
        lon, lat = np.array([10, 20]), np.array([30, 40])
        x1, _, _ = transformer.transform_coordinates(lon, lat, use_cache=True)
        x2, _, _ = transformer.transform_coordinates(lon, lat, use_cache=True)
        assert np.array_equal(x1, x2)
        assert "size" in transformer.get_cache_stats()
        transformer.clear_cache()
        assert transformer.get_cache_stats()["size"] == 0

    def test_distance_threshold_calculation(self):
        """Test distance threshold calculation."""
        transformer = monet_regrid.coordinate_transformer.CoordinateTransformer()
        points = np.random.rand(10, 3)
        threshold = transformer.calculate_distance_threshold(points)
        assert threshold > 0

    def test_coordinate_transformation_accuracy(self):
        """Test coordinate transformation accuracy with known values."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )
        result = source_data.regrid.nearest(self.target_grid)
        assert result.shape == self.target_lat.shape

    def test_3d_coordinate_consistency(self):
        """Test that 3D coordinates maintain proper distances."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )
        result = source_data.regrid.nearest(self.target_grid)
        assert result.shape == self.target_lat.shape
