"""
Numba-optimized kernels for curvilinear interpolation.

This module provides JIT-compiled functions for performing the interpolation loops.
These functions are designed to be used inside xr.apply_ufunc.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_linear(
    data_flat: np.ndarray,
    simplex_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    valid_points: np.ndarray,
    simplex_vertices: np.ndarray,
    fallback_indices: np.ndarray,
) -> np.ndarray:
    """Apply precomputed barycentric weights to interpolate data.

    Parameters
    ----------
    data_flat : np.ndarray
        2D array of source data (n_samples, n_source_points).
    simplex_indices : np.ndarray
        Array of simplex indices for each target point (n_target_points,).
    barycentric_weights : np.ndarray
        Array of barycentric weights for each target point (n_target_points, 4).
    valid_points : np.ndarray
        Boolean array indicating if target point is valid (n_target_points,).
    simplex_vertices : np.ndarray
        Array mapping simplex index to 4 vertex indices (n_simplices, 4).
    fallback_indices : np.ndarray
        Array of fallback (nearest neighbor) indices (n_target_points,).

    Returns
    -------
    np.ndarray
        Interpolated data (n_samples, n_target_points).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(1, 10)
    >>> simplex_indices = np.array([0], dtype=np.int32)
    >>> barycentric_weights = np.array([[0.25, 0.25, 0.25, 0.25]])
    >>> valid_points = np.array([True])
    >>> simplex_vertices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    >>> fallback_indices = np.array([-1], dtype=np.int32)
    >>> res = apply_weights_linear(data, simplex_indices, barycentric_weights,
    ...                            valid_points, simplex_vertices, fallback_indices)
    """
    n_samples = data_flat.shape[0]
    n_targets = len(simplex_indices)

    result = np.full((n_samples, n_targets), np.nan, dtype=data_flat.dtype)

    # Iterate over target points (parallel)
    for i in prange(n_targets):
        if not valid_points[i]:
            continue

        simplex_idx = simplex_indices[i]

        if simplex_idx >= 0:
            # Linear interpolation
            # Get the 4 vertex indices for this simplex
            v0 = simplex_vertices[simplex_idx, 0]
            v1 = simplex_vertices[simplex_idx, 1]
            v2 = simplex_vertices[simplex_idx, 2]
            v3 = simplex_vertices[simplex_idx, 3]

            # Get weights
            w0 = barycentric_weights[i, 0]
            w1 = barycentric_weights[i, 1]
            w2 = barycentric_weights[i, 2]
            w3 = barycentric_weights[i, 3]

            # Interpolate for all samples (inner loop)
            for s in range(n_samples):
                val0 = data_flat[s, v0]
                val1 = data_flat[s, v1]
                val2 = data_flat[s, v2]
                val3 = data_flat[s, v3]

                # Check for NaNs in source data
                if np.isnan(val0) or np.isnan(val1) or np.isnan(val2) or np.isnan(val3):
                    # If any vertex is NaN, use fallback if available
                    if fallback_indices[i] != -1:
                        fallback_idx = fallback_indices[i]
                        result[s, i] = data_flat[s, fallback_idx]
                    else:
                        result[s, i] = np.nan
                else:
                    result[s, i] = w0 * val0 + w1 * val1 + w2 * val2 + w3 * val3

        elif simplex_idx == -2:
            # Nearest neighbor fallback (e.g. outside hull)
            fallback_idx = fallback_indices[i]
            if fallback_idx != -1:
                for s in range(n_samples):
                    result[s, i] = data_flat[s, fallback_idx]

    return result


@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_nearest(
    data_flat: np.ndarray,
    source_indices: np.ndarray,
    valid_points: np.ndarray,
) -> np.ndarray:
    """Apply nearest neighbor interpolation.

    Parameters
    ----------
    data_flat : np.ndarray
        2D array of source data (n_samples, n_source_points).
    source_indices : np.ndarray
        Array of nearest source indices for each target point (n_target_points,).
    valid_points : np.ndarray
        Boolean mask of valid points (n_target_points,).

    Returns
    -------
    np.ndarray
        Interpolated data (n_samples, n_target_points).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[10.0, 20.0]])
    >>> indices = np.array([0, 1], dtype=np.int32)
    >>> valid = np.array([True, True])
    >>> apply_weights_nearest(data, indices, valid)
    array([[10., 20.]])
    """
    n_samples = data_flat.shape[0]
    n_targets = len(source_indices)

    result = np.full((n_samples, n_targets), np.nan, dtype=data_flat.dtype)

    for i in prange(n_targets):
        if not valid_points[i]:
            continue

        idx = source_indices[i]

        for s in range(n_samples):
            result[s, i] = data_flat[s, idx]

    return result


@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_conservative(
    data_flat: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    weights: np.ndarray,
    n_targets: int,
) -> np.ndarray:
    """Apply conservative regridding weights using sparse COO format.

    Parameters
    ----------
    data_flat : np.ndarray
        2D array of source data (n_samples, n_source_points).
    source_indices : np.ndarray
        Indices of source cells (n_interactions,).
    target_indices : np.ndarray
        Indices of target cells (n_interactions,).
    weights : np.ndarray
        Weights for each interaction (n_interactions,).
    n_targets : int
        Number of target cells (to size the output).

    Returns
    -------
    np.ndarray
        Regridded data (n_samples, n_targets).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0]])
    >>> src_idx = np.array([0, 1], dtype=np.int32)
    >>> tgt_idx = np.array([0, 0], dtype=np.int32)
    >>> weights = np.array([0.5, 0.5])
    >>> apply_weights_conservative(data, src_idx, tgt_idx, weights, 1)
    array([[1.5]])
    """
    n_samples = data_flat.shape[0]
    n_interactions = len(weights)

    result = np.zeros((n_samples, n_targets), dtype=data_flat.dtype)

    # Parallelize over samples for maximum speed when n_samples > 1
    if n_samples > 1:
        for s in prange(n_samples):
            for k in range(n_interactions):
                t_idx = target_indices[k]
                s_idx = source_indices[k]
                w = weights[k]

                val = data_flat[s, s_idx]
                if not np.isnan(val):
                    result[s, t_idx] += val * w
    else:
        # Serial execution for n_samples == 1 to avoid race conditions on t_idx
        for k in range(n_interactions):
            t_idx = target_indices[k]
            s_idx = source_indices[k]
            w = weights[k]

            val = data_flat[0, s_idx]
            if not np.isnan(val):
                result[0, t_idx] += val * w

    return result


@jit(nopython=True, nogil=True)
def inverse_bilinear_interpolation(
    p: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> tuple[float, float]:
    """Find local coordinates (u, v) for a point p inside a quadrilateral.

    Solves for p = (1-u)(1-v)v1 + u(1-v)v2 + uvv3 + (1-u)v v4
    using Newton-Raphson.

    Parameters
    ----------
    p : np.ndarray
        (2,) target point.
    v1, v2, v3, v4 : np.ndarray
        (2,) vertices (SW, SE, NE, NW).
    max_iter : int, optional
        Maximum number of iterations. Defaults to 10.
    tol : float, optional
        Convergence tolerance. Defaults to 1e-5.

    Returns
    -------
    tuple[float, float]
        (u, v) where 0 <= u, v <= 1 if inside.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.5, 0.5])
    >>> v1, v2, v3, v4 = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([0,1])
    >>> inverse_bilinear_interpolation(p, v1, v2, v3, v4)
    (0.5, 0.5)
    """
    # Initial guess (center)
    u = 0.5
    v = 0.5

    for _ in range(max_iter):
        # Calculate residuals
        a_var = v1
        b_var = v2 - v1
        c_var = v4 - v1
        d_var = v1 - v2 + v3 - v4

        p_est = a_var + u * b_var + v * c_var + u * v * d_var
        resid = p_est - p

        if np.dot(resid, resid) < tol**2:
            break

        # Jacobian
        j00 = b_var[0] + v * d_var[0]
        j01 = c_var[0] + u * d_var[0]
        j10 = b_var[1] + v * d_var[1]
        j11 = c_var[1] + u * d_var[1]

        det = j00 * j11 - j01 * j10

        if abs(det) < 1e-12:
            break  # Singular, degenerate quad

        inv_det = 1.0 / det
        du = (j11 * resid[0] - j01 * resid[1]) * inv_det
        dv = (j00 * resid[1] - j10 * resid[0]) * inv_det

        u -= du
        v -= dv

    return u, v


@jit(nopython=True, nogil=True)
def _det3x3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Determinant of 3x3 matrix formed by 3 vectors (scalar triple product).

    Parameters
    ----------
    a, b, c : np.ndarray
        (3,) vectors.

    Returns
    -------
    float
        Determinant.

    Examples
    --------
    >>> import numpy as np
    >>> a, b, c = np.eye(3)
    >>> _det3x3(a, b, c)
    1.0
    """
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


@jit(nopython=True, nogil=True, parallel=True)
def compute_linear_weights_grid(
    target_points: np.ndarray,
    source_points: np.ndarray,
    nearest_indices: np.ndarray,
    source_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear interpolation weights using grid-based search.

    Splits each quad into 2 triangles and uses 3D barycentric weights for
    accurate surface interpolation on a sphere.

    Parameters
    ----------
    target_points : np.ndarray
        Array of 3D target points (n_targets, 3).
    source_points : np.ndarray
        Flattened array of 3D source points (n_source, 3).
    nearest_indices : np.ndarray
        Indices of nearest source neighbors for each target point (n_targets,).
    source_shape : tuple[int, int]
        Original shape of the source grid (ny, nx).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices, weights, and valid mask.

    Examples
    --------
    >>> import numpy as np
    >>> src = np.array([[0,0,1], [1,0,1], [1,1,1], [0,1,1]], dtype=np.float64)
    >>> tgt = np.array([[0.5, 0.5, 1.0]])
    >>> near = np.array([0], dtype=np.int32)
    >>> compute_linear_weights_grid(tgt, src, near, (2, 2))
    (array([[0, 2, 3, -1]]), array([[0.5, 0.5, 0.0, 0.0]]), array([True]))
    """
    n_targets = target_points.shape[0]
    ny, nx = source_shape

    out_indices = np.full((n_targets, 4), -1, dtype=np.int32)
    out_weights = np.zeros((n_targets, 4), dtype=np.float64)
    out_valid = np.zeros(n_targets, dtype=np.bool_)

    for k in prange(n_targets):
        nearest_idx = nearest_indices[k]
        j_n = nearest_idx // nx
        i_n = nearest_idx % nx

        p = target_points[k]
        best_sum_pos_w = -1.0
        best_indices = np.array([-1, -1, -1])
        best_weights = np.array([0.0, 0.0, 0.0])

        # Search neighborhood
        for dj in range(-1, 1):
            for di in range(-1, 1):
                j = j_n + dj
                i = i_n + di

                if j < 0 or j >= ny - 1 or i < 0 or i >= nx - 1:
                    continue

                # Quad indices: (j,i), (j,i+1), (j+1,i+1), (j+1,i)
                idx0 = j * nx + i
                idx1 = j * nx + i + 1
                idx2 = (j + 1) * nx + i + 1
                idx3 = (j + 1) * nx + i

                # Split into 2 triangles: (0,1,2) and (0,2,3)
                for t_idxs in [(idx0, idx1, idx2), (idx0, idx2, idx3)]:
                    v0 = source_points[t_idxs[0]]
                    v1 = source_points[t_idxs[1]]
                    v2 = source_points[t_idxs[2]]

                    det_total = _det3x3(v0, v1, v2)
                    if abs(det_total) < 1e-15:
                        continue

                    w0 = _det3x3(p, v1, v2) / det_total
                    w1 = _det3x3(v0, p, v2) / det_total
                    w2 = _det3x3(v0, v1, p) / det_total
                    w3 = 1.0 - (w0 + w1 + w2)  # Weight for Earth center

                    # Point is inside if all surface weights >= 0
                    if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9 and w3 >= -0.1:
                        sum_surface = w0 + w1 + w2
                        out_indices[k, 0] = t_idxs[0]
                        out_indices[k, 1] = t_idxs[1]
                        out_indices[k, 2] = t_idxs[2]
                        out_weights[k, 0] = w0 / sum_surface
                        out_weights[k, 1] = w1 / sum_surface
                        out_weights[k, 2] = w2 / sum_surface
                        out_valid[k] = True
                        best_sum_pos_w = 2.0  # Force exit
                        break

                    # If not strictly inside, keep track of "best" candidate
                    min_w = min(w0, w1, w2, w3)
                    if min_w > best_sum_pos_w:
                        best_sum_pos_w = min_w
                        best_indices = np.array([t_idxs[0], t_idxs[1], t_idxs[2]])
                        sum_surface = w0 + w1 + w2
                        best_weights = np.array([w0 / sum_surface, w1 / sum_surface, w2 / sum_surface])

                if best_sum_pos_w >= 2.0:
                    break
            if best_sum_pos_w >= 2.0:
                break

        # If no triangle strictly contained the point, use the best candidate if it's close enough
        if not out_valid[k] and best_sum_pos_w > -1e-5:
            out_indices[k, 0:3] = best_indices
            out_weights[k, 0:3] = best_weights
            out_valid[k] = True

    return out_indices, out_weights, out_valid


@jit(nopython=True, nogil=True, parallel=True)
def compute_structured_weights(
    target_points: np.ndarray,
    source_points: np.ndarray,
    nearest_indices: np.ndarray,
    source_shape: tuple[int, int],
    method_enum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weights for structured interpolation (bilinear/cubic).

    Parameters
    ----------
    target_points : np.ndarray
        Array of target points (n_targets, 3) or (n_targets, 2).
    source_points : np.ndarray
        Flattened array of source points (n_source, 3) or (n_source, 2).
    nearest_indices : np.ndarray
        Indices of nearest source neighbors (n_targets,).
    source_shape : tuple[int, int]
        Original shape of the source grid (ny, nx).
    method_enum : int
        0 for bilinear, 1 for cubic.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices, weights, and valid mask.

    Examples
    --------
    >>> import numpy as np
    >>> src = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float64)
    >>> tgt = np.array([[0.5, 0.5, 0.0]])
    >>> near = np.array([0], dtype=np.int32)
    >>> compute_structured_weights(tgt, src, near, (2, 2), 0)
    (array([[0, 1, 2, 3]]), array([[0.25, 0.25, 0.25, 0.25]]), array([True]))
    """
    n_targets = target_points.shape[0]
    ny, nx = source_shape

    # Output structure
    max_weights = 16 if method_enum == 1 else 4

    out_indices = np.full((n_targets, max_weights), -1, dtype=np.int32)
    out_weights = np.zeros((n_targets, max_weights), dtype=np.float64)
    out_valid = np.zeros(n_targets, dtype=np.bool_)

    for k in prange(n_targets):
        # 1. Start from nearest neighbor
        nearest_idx = nearest_indices[k]
        j_n = nearest_idx // nx
        i_n = nearest_idx % nx

        found = False
        final_u = 0.0
        final_v = 0.0
        base_j = 0
        base_i = 0

        p = target_points[k, :2]  # Assume projected/2D

        # Search neighborhood (j_n-1 to j_n, i_n-1 to i_n)
        for dj in range(-1, 1):
            for di in range(-1, 1):
                j = j_n + dj
                i = i_n + di

                if j < 0 or j >= ny - 1 or i < 0 or i >= nx - 1:
                    continue

                # Vertices
                idx1 = j * nx + i
                idx2 = j * nx + (i + 1)
                idx3 = (j + 1) * nx + (i + 1)
                idx4 = (j + 1) * nx + i

                v1 = source_points[idx1, :2]
                v2 = source_points[idx2, :2]
                v3 = source_points[idx3, :2]
                v4 = source_points[idx4, :2]

                u, v = inverse_bilinear_interpolation(p, v1, v2, v3, v4)

                # Check if inside [0, 1] with tolerance
                tol = 1e-4
                if -tol <= u <= 1 + tol and -tol <= v <= 1 + tol:
                    found = True
                    final_u = min(max(u, 0.0), 1.0)
                    final_v = min(max(v, 0.0), 1.0)
                    base_j = j
                    base_i = i
                    break
            if found:
                break

        if not found:
            continue

        out_valid[k] = True

        if method_enum == 0:  # Bilinear
            # SW
            out_indices[k, 0] = base_j * nx + base_i
            out_weights[k, 0] = (1 - final_u) * (1 - final_v)
            # SE
            out_indices[k, 1] = base_j * nx + (base_i + 1)
            out_weights[k, 1] = final_u * (1 - final_v)
            # NE
            out_indices[k, 2] = (base_j + 1) * nx + (base_i + 1)
            out_weights[k, 2] = final_u * final_v
            # NW
            out_indices[k, 3] = (base_j + 1) * nx + base_i
            out_weights[k, 3] = (1 - final_u) * final_v

        elif method_enum == 1:  # Cubic
            u = final_u
            v = final_v

            wu0 = -0.5 * u**3 + u**2 - 0.5 * u
            wu1 = 1.5 * u**3 - 2.5 * u**2 + 1.0
            wu2 = -1.5 * u**3 + 2.0 * u**2 + 0.5 * u
            wu3 = 0.5 * u**3 - 0.5 * u**2

            wv0 = -0.5 * v**3 + v**2 - 0.5 * v
            wv1 = 1.5 * v**3 - 2.5 * v**2 + 1.0
            wv2 = -1.5 * v**3 + 2.0 * v**2 + 0.5 * v
            wv3 = 0.5 * v**3 - 0.5 * v**2

            wu = np.array([wu0, wu1, wu2, wu3])
            wv = np.array([wv0, wv1, wv2, wv3])

            count = 0
            for dy in range(-1, 3):
                for dx in range(-1, 3):
                    cur_j = base_j + dy
                    cur_i = base_i + dx

                    cur_j_clamped = min(max(cur_j, 0), ny - 1)
                    cur_i_clamped = min(max(cur_i, 0), nx - 1)

                    idx = cur_j_clamped * nx + cur_i_clamped
                    weight = wv[dy + 1] * wu[dx + 1]

                    out_indices[k, count] = idx
                    out_weights[k, count] = weight
                    count += 1

    return out_indices, out_weights, out_valid


@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_structured(
    data_flat: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Apply structured interpolation weights (bilinear/cubic).

    Parameters
    ----------
    data_flat : np.ndarray
        2D array of source data (n_samples, n_source_points).
    indices : np.ndarray
        Array of indices into source data (n_targets, max_weights).
    weights : np.ndarray
        Array of weights (n_targets, max_weights).
    valid_mask : np.ndarray
        Boolean mask of valid points (n_targets,).

    Returns
    -------
    np.ndarray
        Interpolated data (n_samples, n_targets).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[10, 20, 30, 40]], dtype=np.float64)
    >>> indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    >>> weights = np.array([[0.25, 0.25, 0.25, 0.25]])
    >>> valid = np.array([True])
    >>> apply_weights_structured(data, indices, weights, valid)
    array([[25.]])
    """
    n_samples = data_flat.shape[0]
    n_targets = indices.shape[0]
    max_weights = indices.shape[1]

    result = np.full((n_samples, n_targets), np.nan, dtype=data_flat.dtype)

    for i in prange(n_targets):
        if not valid_mask[i]:
            continue

        for s in range(n_samples):
            val_sum = 0.0
            has_nan = False

            for k in range(max_weights):
                idx = indices[i, k]
                w = weights[i, k]
                val = data_flat[s, idx]
                if np.isnan(val):
                    has_nan = True
                    break
                val_sum += val * w

            if has_nan:
                result[s, i] = np.nan
            else:
                result[s, i] = val_sum

    return result
