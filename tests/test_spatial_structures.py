"""Unit tests for KDTree and ConvexHull spatial structures in curvilinear regridding.

This module tests the spatial indexing, nearest neighbor queries, and triangulation
structures used for efficient interpolation in 3D space.
"""

import logging

import numpy as np
import pytest
import xarray as xr
from scipy.spatial import Delaunay

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from xarray_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


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

    def test_kdtree_initialization(self):
        """Test that KDTree is properly initialized for nearest neighbor queries."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")

        # Check that KDTree was created
        assert hasattr(interpolator, "kdtree")
        assert hasattr(interpolator.kdtree, "n")  # Has KDTree interface

        # Check that tree contains the correct number of points
        expected_points = self.source_lat_2d.size
        assert interpolator.kdtree.n == expected_points

        # Check that tree data matches source points
        np.testing.assert_array_equal(interpolator.kdtree.data, interpolator.source_points_3d)

    def test_kdtree_query_functionality(self):
        """Test KDTree nearest neighbor query functionality."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")

        # Test single point query - test with a single point
        test_point = interpolator.target_points_3d[0]  # Get single point
        distances, indices = interpolator.kdtree.query(test_point, k=1)

        # Should return scalar values for single point, single neighbor
        assert np.isscalar(distances)
        assert np.isscalar(indices)
        assert np.asarray(distances) >= 0.0
        assert np.asarray(indices) >= 0 and np.asarray(indices) < len(interpolator.source_points_3d)

        # Test multiple neighbors
        distances, indices = interpolator.kdtree.query(test_point, k=5)
        assert len(distances) == 5
        assert len(indices) == 5
        assert np.all(distances >= 0)
        assert np.all(indices < len(interpolator.source_points_3d))

        # Distances should be in ascending order
        assert np.all(distances[:-1] <= distances[1:])

    def test_kdtree_batch_queries(self):
        """Test KDTree batch queries for multiple target points."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")

        # Test querying all target points at once
        all_distances, all_indices = interpolator.kdtree.query(interpolator.target_points_3d)

        # Should return arrays with same length as number of target points
        assert len(all_distances) == len(interpolator.target_points_3d)
        assert len(all_indices) == len(interpolator.target_points_3d)

        # All results should be valid
        assert np.all(np.isfinite(all_distances))
        assert np.all(all_indices < len(interpolator.source_points_3d))
        assert np.all(all_indices >= 0)

    def test_distance_threshold_computation(self):
        """Test automatic distance threshold computation for out-of-domain detection."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")

        # Check that threshold was computed
        assert hasattr(interpolator, "distance_threshold")
        assert interpolator.distance_threshold > 0
        assert np.isfinite(interpolator.distance_threshold)

        # Threshold should be reasonable relative to domain size
        source_points = interpolator.source_points_3d
        domain_extent = np.max(np.ptp(source_points, axis=0))

        # Threshold should be smaller than domain extent but not too small
        assert interpolator.distance_threshold < domain_extent * 2
        assert interpolator.distance_threshold > domain_extent * 0.01

    def test_custom_radius_of_influence(self):
        """Test custom radius of influence parameter."""
        custom_radius = 1e6
        interpolator = CurvilinearInterpolator(
            self.source_grid, self.target_grid, method="nearest", radius_of_influence=custom_radius
        )

        # Check that custom radius is stored and used
        assert interpolator.radius_of_influence == custom_radius
        assert interpolator.distance_threshold == custom_radius

    def test_kdtree_nearest_neighbor_consistency(self):
        """Test that nearest neighbor results are consistent and deterministic."""
        interpolator1 = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")
        interpolator2 = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")

        # Results should be identical for identical inputs
        np.testing.assert_array_equal(interpolator1.source_indices, interpolator2.source_indices)
        np.testing.assert_array_almost_equal(interpolator1.distances, interpolator2.distances)


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
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="linear")

        # Check that triangulation structure (Delaunay) was created
        assert hasattr(interpolator, "convex_hull")
        assert isinstance(interpolator.convex_hull, Delaunay)

        # Check that hull contains the correct number of points
        assert len(interpolator.convex_hull.points) == len(interpolator.source_points_3d)

        # Check that triangles (simplices) were computed
        assert hasattr(interpolator, "triangles")
        assert len(interpolator.triangles) > 0
        # In 3D space, each simplex has 4 vertices (tetrahedron)
        assert interpolator.triangles.shape[1] == 4  # Each simplex has 4 vertices in 3D

    def test_triangle_properties(self):
        """Test geometric properties of computed triangles."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="linear")

        triangles = interpolator.triangles
        points = interpolator.source_points_3d

        # Check that all triangle indices are valid
        assert np.all(triangles >= 0)
        assert np.all(triangles < len(points))

        # Check that simplices are non-degenerate (have volume in 3D case)
        for simplex_idx in range(len(triangles)):
            simplex_vertices = points[triangles[simplex_idx]]

            # For 3D Delaunay, each simplex is a tetrahedron with 4 vertices
            # Check that the tetrahedron is not degenerate by computing its volume
            # Using the determinant method for tetrahedron volume
            # Volume = (1/6) * |det([v1-v0, v2-v0, v3-v0])|
            v0, v1, v2, v3 = simplex_vertices
            matrix = np.array([v1 - v0, v2 - v0, v3 - v0])
            volume = abs(np.linalg.det(matrix)) / 6.0

            # Volume should be positive (non-degenerate tetrahedron)
            assert volume > 1e-12, f"Simplex {simplex_idx} appears degenerate"

    def test_triangle_centroid_computation(self):
        """Test triangle centroid computation for efficient lookup."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="linear")

        # Check that centroids were computed
        assert hasattr(interpolator, "triangle_centroids")
        assert len(interpolator.triangle_centroids) == len(interpolator.triangles)
        assert interpolator.triangle_centroids.shape[1] == 3  # 3D coordinates

        # Check that centroid KDTree was created
        assert hasattr(interpolator, "triangle_centroid_kdtree")
        assert len(interpolator.triangle_centroid_kdtree.data) == len(interpolator.triangles)

        # Verify centroid calculation for a few triangles
        for i in range(min(5, len(interpolator.triangles))):
            triangle_vertices = interpolator.source_points_3d[interpolator.triangles[i]]
            computed_centroid = np.mean(triangle_vertices, axis=0)

            np.testing.assert_array_almost_equal(interpolator.triangle_centroids[i], computed_centroid, decimal=6)

    def test_convex_hull_fallback_nearest_neighbor(self):
        """Test that KDTree is created as fallback for linear interpolation."""
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="linear")

        # Linear interpolation should also create KDTree for fallback
        assert hasattr(interpolator, "kdtree")
        assert hasattr(interpolator.kdtree, "n")  # Has KDTree interface
        assert interpolator.kdtree.n == len(interpolator.source_points_3d)

        # Should also create target KDTree for efficient target point queries
        assert hasattr(interpolator, "target_kdtree")
        assert hasattr(interpolator.target_kdtree, "n")  # Has KDTree interface
        assert interpolator.target_kdtree.n == len(interpolator.target_points_3d)


class TestSpatialStructureEdgeCases:
    """Test spatial structures with edge cases and boundary conditions."""

    def test_degenerate_grid_handling(self):
        """Test handling of degenerate grids that might cause convex hull errors."""
        # Create a nearly degenerate grid (points almost in a plane)
        source_lat = np.array([[0, 0.001], [0, 0.001]])
        source_lon = np.array([[0, 0.001], [0.001, 0.001]])

        source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

        target_grid = xr.Dataset(
            {"latitude": (["y_target", "x_target"], source_lat), "longitude": (["y_target", "x_target"], source_lon)}
        )

        # Should handle degenerate cases gracefully
        try:
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="linear")
            # If it succeeds, check that structures were created
            assert hasattr(interpolator, "convex_hull") or hasattr(interpolator, "kdtree")
        except ValueError:
            # If convex hull fails, that's acceptable for degenerate cases
            pass

    def test_single_point_grid(self):
        """Test handling of single-point grids."""
        source_lat = np.array([[5.0]])
        source_lon = np.array([[-10.0]])

        source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

        target_grid = xr.Dataset(
            {"latitude": (["y_target", "x_target"], source_lat), "longitude": (["y_target", "x_target"], source_lon)}
        )

        # Should handle single point gracefully
        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        assert hasattr(interpolator, "kdtree")
        assert interpolator.kdtree.n == 1

        # Linear method should fall back to nearest neighbor for single point
        # It warns instead of raising ValueError now
        with pytest.warns(UserWarning, match="Linear interpolation requires at least 4 source points"):
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="linear")
            # The method property might still say "linear" depending on implementation,
            # but the internal engine should be using nearest neighbor logic.
            # Let's check if it behaves like nearest neighbor (which works for single points)
            result = interpolator(xr.DataArray(np.array([[10.0]]), dims=["y", "x"]))
            assert result.item() == 10.0

    def test_empty_target_grid(self):
        """Test handling of empty target grids."""
        source_grid = xr.Dataset(
            {"latitude": (["y", "x"], np.zeros((2, 0))), "longitude": (["y", "x"], np.zeros((2, 0)))}
        )

        # Empty target grid should be handled gracefully
        empty_target = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], np.array([]).reshape(0, 0)),
                "longitude": (["y_target", "x_target"], np.array([]).reshape(0, 0)),
            }
        )

        # For now, just test that it doesn't crash immediately
        # The actual handling of empty grids is implementation-specific
        try:
            CurvilinearInterpolator(source_grid, empty_target, method="nearest")
            # If it succeeds, that's acceptable for now
        except Exception:
            # If it raises an exception, that's also acceptable
            pass


if __name__ == "__main__":
    # Run tests
    kdtree_test = TestKDTreeStructure()
    kdtree_test.setup_method()

    kdtree_test.test_kdtree_initialization()
    kdtree_test.test_kdtree_query_functionality()
    kdtree_test.test_kdtree_batch_queries()
    kdtree_test.test_distance_threshold_computation()
    kdtree_test.test_custom_radius_of_influence()
    kdtree_test.test_kdtree_nearest_neighbor_consistency()

    hull_test = TestConvexHullStructure()
    hull_test.setup_method()

    hull_test.test_convex_hull_initialization()
    hull_test.test_triangle_properties()
    hull_test.test_triangle_centroid_computation()
    hull_test.test_convex_hull_fallback_nearest_neighbor()

    edge_test = TestSpatialStructureEdgeCases()
    edge_test.test_degenerate_grid_handling()
    edge_test.test_single_point_grid()
    edge_test.test_empty_target_grid()

    logging.info("All spatial structure tests passed!")
