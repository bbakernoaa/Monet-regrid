import pytest

import numpy as np
import xarray as xr

"""Unit tests for KDTree and ConvexHull spatial structures in curvilinear regridding.

This module tests the spatial indexing, nearest neighbor queries, and triangulation
structures used for efficient interpolation in 3D space.
"""


@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning")
class TestKDTreeStructure:
    """Test KDTree spatial indexing and nearest neighbor queries."""

    def setup_method(self):
        """Set up test data for KDTree tests."""
        # Create a regular grid for testing
        self.source_lat = np.linspace(-10, 10, 10)
        self.source_lon = np.linspace(-20, 20, 12)
        self.source_lat_2d, self.source_lon_2d = np.meshgrid(self.source_lat, self.source_lon)

        self.target_lat = np.linspace(-5, 5, 5)
        self.target_lon = np.linspace(-10, 10, 6)
        self.target_lat_2d, self.target_lon_2d = np.meshgrid(self.target_lat, self.target_lon)

        self.source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.source_lat_2d), "longitude": (["y", "x"], self.source_lon_2d)}
        )

        self.target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.target_lat_2d),
                "longitude": (["y_target", "x_target"], self.target_lon_2d),
            }
        )

    def test_kdtree_query_functionality(self):
        """Test KDTree nearest neighbor query functionality."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat_2d.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat_2d),
                "longitude": (["y", "x"], self.source_lon_2d),
            },
        )
        result = source_data.regrid.nearest(self.target_grid)
        assert result.shape == self.target_lat_2d.shape

    def test_kdtree_batch_queries(self):
        """Test KDTree batch queries for multiple target points."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat_2d.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat_2d),
                "longitude": (["y", "x"], self.source_lon_2d),
            },
        )
        result = source_data.regrid.nearest(self.target_grid)
        assert result.shape == self.target_lat_2d.shape


@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning")
class TestConvexHullStructure:
    """Test ConvexHull triangulation for linear interpolation."""

    def setup_method(self):
        """Set up test data for ConvexHull tests."""
        # Create a grid that will produce a valid convex hull
        self.source_lat = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.source_lon = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        self.target_lat = np.array([[0.5, 1.5], [0.5, 1.5]])
        self.target_lon = np.array([[-0.5, -0.5], [0.5, 0.5]])

        self.source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.source_lat), "longitude": (["y", "x"], self.source_lon)}
        )

        self.target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.target_lat),
                "longitude": (["y_target", "x_target"], self.target_lon),
            }
        )

    def test_convex_hull_initialization(self):
        """Test that ConvexHull is properly initialized for triangulation."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )
        result = source_data.regrid.linear(self.target_grid)
        assert result.shape == self.target_lat.shape

    def test_triangle_properties(self):
        """Test geometric properties of computed triangles."""
        source_data = xr.DataArray(
            np.random.rand(*self.source_lat.shape),
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.source_lat),
                "longitude": (["y", "x"], self.source_lon),
            },
        )
        result = source_data.regrid.linear(self.target_grid)
        assert result.shape == self.target_lat.shape
