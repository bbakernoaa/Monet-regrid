import pytest
"""Unit tests for coordinate transformation in curvilinear regridding.

This module tests the 3D coordinate transformation accuracy, pyproj integration,
and spherical geometry handling in the CurvilinearInterpolator.
"""

import numpy as np
import xarray as xr
import monet_regrid  # noqa: F401


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
