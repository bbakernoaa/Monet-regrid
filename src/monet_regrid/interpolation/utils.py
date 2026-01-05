"""
Utility functions for interpolation.
"""

import numpy as np

try:
    from monet_regrid.methods._numba_kernels import (
        inverse_bilinear_interpolation,
    )

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    inverse_bilinear_interpolation = None

__all__ = [
    "HAS_NUMBA",
    "_compute_barycentric_weights_3d",
    "_point_in_tetrahedron",
    "inverse_bilinear_interpolation",
]


def _point_in_tetrahedron(point: np.ndarray, tetra_vertices: np.ndarray) -> bool:
    """Check if a 3D point is contained in a tetrahedron."""
    weights = _compute_barycentric_weights_3d(point, tetra_vertices)
    return bool(weights is not None and np.all(weights >= -1e-9) and np.all(weights <= 1 + 1e-9))


def _compute_barycentric_weights_3d(point: np.ndarray, tetra_vertices: np.ndarray) -> np.ndarray | None:
    """Compute barycentric weights for a point in a 3D tetrahedron."""
    # Using the matrix inversion method
    t_matrix = np.vstack((tetra_vertices.T, np.ones(4)))
    p = np.append(point, 1)
    try:
        weights = np.linalg.solve(t_matrix, p)
        return weights
    except np.linalg.LinAlgError:
        # Singular matrix, likely degenerate tetrahedron
        return None


def _point_in_triangle_3d(point: np.ndarray, triangle_vertices: np.ndarray) -> bool:
    """Check if a 3D point is contained in a triangle using barycentric coordinates."""
    # Legacy method
    v0 = triangle_vertices[1] - triangle_vertices[0]
    v1 = triangle_vertices[2] - triangle_vertices[0]
    v2 = point - triangle_vertices[0]

    # Calculate dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Calculate barycentric coordinates
    try:
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    except ZeroDivisionError:
        return False  # Degenerate triangle

    return bool((u >= 0) and (v >= 0) and (u + v <= 1))


def _compute_barycentric_weights_2d_in_3d(point: np.ndarray, triangle_vertices: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for a point in a 3D triangle."""
    # Legacy method
    v0 = triangle_vertices[1] - triangle_vertices[0]
    v1 = triangle_vertices[2] - triangle_vertices[0]
    v2 = point - triangle_vertices[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    try:
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1 - u - v  # Weight for first vertex
    except ZeroDivisionError:
        return np.array([1 / 3.0, 1 / 3.0, 1 / 3.0])

    return np.array([w, u, v])
