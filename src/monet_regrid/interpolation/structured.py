"""
Optimized grid search and weight calculation for structured grids using Numba.

This module provides a high-performance, two-stage function to compute
interpolation weights for linear interpolation on structured curvilinear grids.
It combines a highly optimized KD-tree for an initial guess with a localized,
Numba-accelerated grid search to find the containing cell and compute weights.
This avoids the high cost of global Delaunay triangulation and brute-force
searches.

This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 [Your Organization].
"""
import numba
import numpy as np
from scipy.spatial import cKDTree


@numba.jit(nopython=True, cache=True)
def _point_in_triangle_2d(p, p0, p1, p2):
    """Check if a point is inside a triangle in 2D using barycentric coordinates."""
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = p - p0
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v < 1)


@numba.jit(nopython=True, cache=True)
def _search_and_compute_weights(
    source_points_3d: np.ndarray,
    target_points_3d: np.ndarray,
    nearest_indices: np.ndarray,
    source_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a localized search and compute barycentric weights."""
    n_target = target_points_3d.shape[0]
    n_source_lat, n_source_lon = source_shape
    distances = np.full(n_target, np.nan, dtype=np.float64)
    source_indices = np.full((n_target, 4), -1, dtype=np.int64)
    weights = np.full((n_target, 4), np.nan, dtype=np.float64)

    for i in range(n_target):
        target_point = target_points_3d[i]
        best_j = nearest_indices[i] // n_source_lon
        best_k = nearest_indices[i] % n_source_lon

        found = False
        for j_offset in range(-2, 2):
            for k_offset in range(-2, 2):
                j_start, k_start = best_j + j_offset, best_k + k_offset
                if 0 <= j_start < n_source_lat - 1 and 0 <= k_start < n_source_lon - 1:
                    p00_idx = j_start * n_source_lon + k_start
                    p01_idx = (j_start) * n_source_lon + (k_start + 1)
                    p10_idx = (j_start + 1) * n_source_lon + k_start
                    p11_idx = (j_start + 1) * n_source_lon + (k_start + 1)
                    p00, p01, p10, p11 = (
                        source_points_3d[p00_idx],
                        source_points_3d[p01_idx],
                        source_points_3d[p10_idx],
                        source_points_3d[p11_idx],
                    )
                    normal = np.cross(p10 - p00, p01 - p00)
                    p_2d, p00_2d, p01_2d, p10_2d, p11_2d = (
                        np.empty(2, dtype=np.float64),
                        np.empty(2, dtype=np.float64),
                        np.empty(2, dtype=np.float64),
                        np.empty(2, dtype=np.float64),
                        np.empty(2, dtype=np.float64),
                    )

                    if np.abs(normal[0]) > np.abs(normal[1]) and np.abs(normal[0]) > np.abs(normal[2]):
                        p_2d[0], p_2d[1] = target_point[1], target_point[2]
                        p00_2d[0], p00_2d[1] = p00[1], p00[2]
                        p01_2d[0], p01_2d[1] = p01[1], p01[2]
                        p10_2d[0], p10_2d[1] = p10[1], p10[2]
                        p11_2d[0], p11_2d[1] = p11[1], p11[2]
                    elif np.abs(normal[1]) > np.abs(normal[2]):
                        p_2d[0], p_2d[1] = target_point[0], target_point[2]
                        p00_2d[0], p00_2d[1] = p00[0], p00[2]
                        p01_2d[0], p01_2d[1] = p01[0], p01[2]
                        p10_2d[0], p10_2d[1] = p10[0], p10[2]
                        p11_2d[0], p11_2d[1] = p11[0], p11[2]
                    else:
                        p_2d[0], p_2d[1] = target_point[0], target_point[1]
                        p00_2d[0], p00_2d[1] = p00[0], p00[1]
                        p01_2d[0], p01_2d[1] = p01[0], p01[1]
                        p10_2d[0], p10_2d[1] = p10[0], p10[1]
                        p11_2d[0], p11_2d[1] = p11[0], p11[1]

                    if _point_in_triangle_2d(p_2d, p00_2d, p01_2d, p10_2d):
                        v0, v1, v2 = p01 - p00, p10 - p00, target_point - p00
                        d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
                        denom = d00 * d11 - d01 * d01
                        if np.abs(denom) > 1e-12:
                            w1, w2 = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
                            weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3] = 1.0 - w1 - w2, w1, w2, 0.0
                            source_indices[i, 0], source_indices[i, 1], source_indices[i, 2] = p00_idx, p01_idx, p10_idx
                            found = True
                            break
                    if _point_in_triangle_2d(p_2d, p11_2d, p10_2d, p01_2d):
                        v0, v1, v2 = p10 - p11, p01 - p11, target_point - p11
                        d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
                        denom = d00 * d11 - d01 * d01
                        if np.abs(denom) > 1e-12:
                            w1, w2 = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
                            weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3] = 1.0 - w1 - w2, w1, w2, 0.0
                            source_indices[i, 0], source_indices[i, 1], source_indices[i, 2] = p11_idx, p10_idx, p01_idx
                            found = True
                            break
            if found:
                break
    return distances, source_indices, weights


def compute_linear_weights_grid(
    source_points_3d: np.ndarray,
    target_points_3d: np.ndarray,
    source_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute linear interpolation weights for a structured grid.
    This function uses a two-stage approach: a KD-tree for an initial guess
    and a Numba-accelerated local search to find the containing cell and
    compute barycentric weights.
    """
    # Stage 1: Use KD-tree to find the nearest source point for each target point.
    tree = cKDTree(source_points_3d)
    _, nearest_indices = tree.query(target_points_3d, k=1)

    # Stage 2: Use Numba-accelerated local search starting from the nearest point.
    distances, source_indices, weights = _search_and_compute_weights(
        source_points_3d, target_points_3d, nearest_indices, source_shape
    )
    return distances, source_indices, weights
