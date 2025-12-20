"""
Tests for spatial data structures (KDTree, etc.) and their integration.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from monet_regrid.interpolation.base import HAS_PYKDTREE
from monet_regrid.interpolation.base import cKDTree as PyKDTree
from monet_regrid.coordinate_transformer import CoordinateTransformer

_transformer = CoordinateTransformer()
_cartesian_to_geographic_2d = _transformer.inverse_transform_coordinates
_geographic_to_cartesian_2d = _transformer.transform_coordinates

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import paths: from monet_regrid.spatial import ...
# New import paths: from monet_regrid.interpolation.base import ...


def test_kdtree_creation_and_query():
    """Test KDTree creation and basic querying."""
    points = np.random.rand(100, 3)  # 100 points in 3D space
    tree = cKDTree(points)

    # Query with a single point
    query_point = np.array([0.5, 0.5, 0.5])
    dist, idx = tree.query(query_point)

    assert isinstance(dist, float)
    assert isinstance(idx, int)
    assert idx < 100

    # Query with multiple points
    query_points = np.random.rand(10, 3)
    dists, idxs = tree.query(query_points)
    assert dists.shape == (10,)
    assert idxs.shape == (10,)


def test_pykdtree_adapter_scipy_equivalence():
    """Test if pykdtree adapter behaves like scipy.spatial.cKDTree."""
    if not HAS_PYKDTREE:
        return

    points = np.random.rand(50, 2)
    query_points = np.random.rand(5, 2)

    # SciPy KDTree
    scipy_tree = cKDTree(points)
    scipy_dists, scipy_idxs = scipy_tree.query(query_points, k=3)

    # PyKDTree Adapter
    pykdtree_tree = PyKDTree(points)
    pykdtree_dists, pykdtree_idxs = pykdtree_tree.query(query_points, k=3)

    # Compare results
    np.testing.assert_allclose(scipy_dists, pykdtree_dists)
    np.testing.assert_array_equal(scipy_idxs, pykdtree_idxs)


def test_coordinate_conversion_roundtrip():
    """Test roundtrip conversion between geographic and Cartesian coordinates."""
    lons = np.array([-180, -90, 0, 90, 180])
    lats = np.array([-90, -45, 0, 45, 90])
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    x, y, z = _geographic_to_cartesian_2d(lon_grid, lat_grid)
    lon_rt, lat_rt, _ = _cartesian_to_geographic_2d(x, y, z)

    # Use larger tolerance for pole points
    np.testing.assert_allclose(lon_grid, lon_rt, atol=1e-5)
    np.testing.assert_allclose(lat_grid, lat_rt, atol=1e-5)


def test_kdtree_on_geographic_coordinates():
    """Test building and querying KDTree on geographic coordinates converted to 3D."""
    # Create geographic coordinates (lon, lat)
    lons = np.linspace(-180, 180, 20)
    lats = np.linspace(-90, 90, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Convert to 3D Cartesian
    x, y, z = _geographic_to_cartesian_2d(lon_grid, lat_grid)
    points_3d = np.array([x.ravel(), y.ravel(), z.ravel()]).T

    # Build KDTree
    tree = cKDTree(points_3d)

    # Query with a point (e.g., lon=10, lat=20)
    query_lon, query_lat = 10, 20
    xq, yq, zq = _geographic_to_cartesian_2d(query_lon, query_lat)
    query_point_3d = np.array([xq, yq, zq])

    _dist, idx = tree.query(query_point_3d)

    # The closest point in the 3D tree should correspond to the closest grid point
    closest_point_3d = points_3d[idx]
    clon, clat, _ = _cartesian_to_geographic_2d(
        closest_point_3d[0], closest_point_3d[1], closest_point_3d[2]
    )

    # Find the true closest grid point by geographic distance (for verification)
    distances_geo = np.sqrt((lon_grid - query_lon) ** 2 + (lat_grid - query_lat) ** 2)
    true_closest_idx = np.argmin(distances_geo)
    true_clon = lon_grid.ravel()[true_closest_idx]
    true_clat = lat_grid.ravel()[true_closest_idx]

    assert np.isclose(clon, true_clon)
    assert np.isclose(clat, true_clat)
