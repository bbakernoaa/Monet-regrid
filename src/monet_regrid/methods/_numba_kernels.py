"""
Numba-optimized kernels for curvilinear interpolation.

This module provides JIT-compiled functions for performing the interpolation loops.
These functions are designed to be used inside xr.apply_ufunc.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_linear(
    data_flat,  # (n_samples, n_source_points)
    simplex_indices,  # (n_target_points,)
    barycentric_weights,  # (n_target_points, 4)
    valid_points,  # (n_target_points,)
    simplex_vertices,  # (n_simplices, 4) - indices into data_flat
    fallback_indices,  # (n_target_points,) - indices into data_flat or -1
):
    """
    Apply precomputed barycentric weights to interpolate data.

    Args:
        data_flat: 2D array of source data (n_samples, n_source_points)
        simplex_indices: Array of simplex indices for each target point
        barycentric_weights: Array of barycentric weights for each target point
        valid_points: Boolean array indicating if target point is valid
        simplex_vertices: Array mapping simplex index to 4 vertex indices
        fallback_indices: Array of fallback (nearest neighbor) indices

    Returns:
        Interpolated data (n_samples, n_target_points)
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
                    # If any vertex is NaN, result is NaN (unless we want to implement fallback here)
                    # The original implementation had a complex fallback here which is hard to replicate exactly in Numba
                    # efficiently without passing more data (KDTree etc).
                    # For now, we leave as NaN to be consistent with standard linear interp behavior,
                    # or users can fillna() before regridding.

                    # However, if fallback_indices are provided, we can use them
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
    data_flat,  # (n_samples, n_source_points)
    source_indices,  # (n_target_points,)
    valid_points,  # (n_target_points,) boolean (e.g. distance < threshold)
):
    """
    Apply nearest neighbor interpolation.

    Args:
        data_flat: 2D array of source data
        source_indices: Array of nearest source indices for each target point
        valid_points: Boolean mask of valid points (distance threshold check)

    Returns:
        Interpolated data
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
    data_flat,  # (n_samples, n_source_points)
    source_indices,  # (n_target_points, max_overlaps)
    weights,  # (n_target_points, max_overlaps)
    valid_mask,  # (n_target_points,) or similar validity check? No, (n_target_points, max_overlaps)
):
    """
    Apply conservative regridding weights.

    Args:
        data_flat: 2D array of source data
        source_indices: Indices of source cells contributing to each target cell
        weights: Weights for each contribution
        valid_mask: Boolean mask indicating valid overlaps (padded entries are False)

    Returns:
        Regridded data
    """
    n_samples = data_flat.shape[0]
    n_targets = source_indices.shape[0]
    max_overlaps = source_indices.shape[1]

    result = np.zeros((n_samples, n_targets), dtype=data_flat.dtype)

    for i in prange(n_targets):
        for k in range(max_overlaps):
            # Check validity
            if not valid_mask[i, k]:
                continue

            s_idx = source_indices[i, k]
            w = weights[i, k]

            for s in range(n_samples):
                val = data_flat[s, s_idx]
                # Conservative regridding typically assumes NaNs are 0 or ignored in sum
                # But strict conservation requires handling them.
                # If we assume 'skipna=True' logic where NaNs don't contribute:
                if not np.isnan(val):
                    result[s, i] += val * w
                # If we wanted to propagate NaN, we'd check if any contributor is NaN -> result NaN
                # But usually conservative is about aggregation.

    return result

@jit(nopython=True, nogil=True)
def inverse_bilinear_interpolation(p, v1, v2, v3, v4, max_iter=10, tol=1e-5):
    """
    Find local coordinates (u, v) for a point p inside a quadrilateral defined by v1, v2, v3, v4.
    Solves for p = (1-u)(1-v)v1 + u(1-v)v2 + uvv3 + (1-u)v v4

    We assume the quad is roughly planar or we project to a local plane?
    Actually, we can solve this in 2D if we just use the first 2 coordinates (x, y)
    assuming the problem is defined in a projected space or lat/lon space.
    If 3D, it's overdetermined but we can minimize distance.

    Standard approach for general quad: Newton-Raphson on 2D coordinates.

    Args:
        p: (2,) target point
        v1, v2, v3, v4: (2,) vertices (SW, SE, NE, NW)

    Returns:
        (u, v) where 0 <= u, v <= 1 if inside.
    """
    # Initial guess (center)
    u = 0.5
    v = 0.5

    for _ in range(max_iter):
        # Calculate residuals
        # p_est = (1-u)(1-v)v1 + u(1-v)v2 + uvv3 + (1-u)v v4
        #       = v1 + u(v2-v1) + v(v4-v1) + uv(v1-v2+v3-v4)

        A = v1
        B = v2 - v1
        C = v4 - v1
        D = v1 - v2 + v3 - v4

        p_est = A + u*B + v*C + u*v*D
        resid = p_est - p

        if np.dot(resid, resid) < tol**2:
            break

        # Jacobian
        # dP/du = B + vD
        # dP/dv = C + uD
        J00 = B[0] + v*D[0]
        J01 = C[0] + u*D[0]
        J10 = B[1] + v*D[1]
        J11 = C[1] + u*D[1]

        det = J00*J11 - J01*J10

        if abs(det) < 1e-12:
            break # Singular, degenerate quad

        inv_det = 1.0 / det
        du = (J11*resid[0] - J01*resid[1]) * inv_det
        dv = (J00*resid[1] - J10*resid[0]) * inv_det

        u -= du
        v -= dv

    return u, v

@jit(nopython=True, nogil=True, parallel=True)
def compute_structured_weights(
    target_points, # (n_targets, 3) or (n_targets, 2)
    source_points, # (n_source, 3) or (n_source, 2) - flattened
    nearest_indices, # (n_targets,) from KDTree
    source_shape, # (ny, nx)
    method_enum, # 0=bilinear, 1=cubic
):
    """
    Compute weights for structured interpolation (bilinear/cubic).

    We assume source_points are flattened from (ny, nx).
    Indices map as: idx = j * nx + i
    """
    n_targets = target_points.shape[0]
    ny, nx = source_shape

    # Output structure
    # For bilinear: 4 weights. For cubic: 16 weights.
    max_weights = 16 if method_enum == 1 else 4

    out_indices = np.full((n_targets, max_weights), -1, dtype=np.int32)
    out_weights = np.zeros((n_targets, max_weights), dtype=np.float64)
    out_valid = np.zeros(n_targets, dtype=np.bool_)

    for k in prange(n_targets):
        # 1. Start from nearest neighbor
        nearest_idx = nearest_indices[k]
        j_n = nearest_idx // nx
        i_n = nearest_idx % nx

        # 2. Check 4 surrounding cells (quadrants) to find which one contains the point
        # A cell (j, i) is formed by (j,i), (j,i+1), (j+1,i+1), (j+1,i)

        found = False
        final_u = 0.0
        final_v = 0.0
        base_j = 0
        base_i = 0

        p = target_points[k, :2] # Assume projected/2D

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
                if -tol <= u <= 1+tol and -tol <= v <= 1+tol:
                    found = True
                    final_u = min(max(u, 0.0), 1.0)
                    final_v = min(max(v, 0.0), 1.0)
                    base_j = j
                    base_i = i
                    break
            if found:
                break

        if not found:
            # Fallback to nearest neighbor or extrapolation?
            # For now, mark invalid or use nearest (which is essentially what we started with)
            # Actually, let's keep it simple: if not found, use nearest (weight 1.0)
            # Or leave valid=False
            continue

        out_valid[k] = True

        if method_enum == 0: # Bilinear
            # Weights: (1-u)(1-v), u(1-v), uv, (1-u)v
            # Indices: (j,i), (j,i+1), (j+1,i+1), (j+1,i)
            # SW, SE, NE, NW

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

        elif method_enum == 1: # Cubic
            # Bicubic interpolation on the index space (u, v)
            # We need 4x4 stencil: from base_i-1 to base_i+2

            # Compute cubic weights for u
            # Catmull-Rom spline or similar?
            # Standard bicubic convolution weights
            # w0(t) = -0.5t^3 + t^2 - 0.5t
            # w1(t) = 1.5t^3 - 2.5t^2 + 1
            # w2(t) = -1.5t^3 + 2t^2 + 0.5t
            # w3(t) = 0.5t^3 - 0.5t^2

            u = final_u
            v = final_v

            wu0 = -0.5*u**3 + u**2 - 0.5*u
            wu1 = 1.5*u**3 - 2.5*u**2 + 1.0
            wu2 = -1.5*u**3 + 2.0*u**2 + 0.5*u
            wu3 = 0.5*u**3 - 0.5*u**2

            wv0 = -0.5*v**3 + v**2 - 0.5*v
            wv1 = 1.5*v**3 - 2.5*v**2 + 1.0
            wv2 = -1.5*v**3 + 2.0*v**2 + 0.5*v
            wv3 = 0.5*v**3 - 0.5*v**2

            wu = np.array([wu0, wu1, wu2, wu3])
            wv = np.array([wv0, wv1, wv2, wv3])

            count = 0
            for dy in range(-1, 3):
                for dx in range(-1, 3):
                    cur_j = base_j + dy
                    cur_i = base_i + dx

                    # Clamp to boundaries (repeat edge pixels)
                    cur_j_clamped = min(max(cur_j, 0), ny - 1)
                    cur_i_clamped = min(max(cur_i, 0), nx - 1)

                    idx = cur_j_clamped * nx + cur_i_clamped
                    weight = wv[dy+1] * wu[dx+1]

                    out_indices[k, count] = idx
                    out_weights[k, count] = weight
                    count += 1

    return out_indices, out_weights, out_valid

@jit(nopython=True, nogil=True, parallel=True)
def apply_weights_structured(
    data_flat,  # (n_samples, n_source_points)
    indices,    # (n_targets, max_weights)
    weights,    # (n_targets, max_weights)
    valid_mask, # (n_targets,)
):
    """
    Apply structured interpolation weights (bilinear/cubic).
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
            weight_sum = 0.0

            for k in range(max_weights):
                idx = indices[i, k]
                w = weights[i, k]

                val = data_flat[s, idx]
                if not np.isnan(val):
                    val_sum += val * w
                    # weight_sum += w # For normalization if needed? Usually sum(w)=1

            # For bicubic, sum(weights) is 1.0 but weights can be negative.
            # If NaNs are present, handling is tricky.
            # Simple approach: if any NaN in stencil, result is NaN (safe).
            # Or ignore NaNs (renormalize).
            # Here we just output the sum. If all valid, it's correct.
            # If some NaN, val_sum will be partial.
            # Let's verify NaNs strictly.

            has_nan = False
            for k in range(max_weights):
                if np.isnan(data_flat[s, indices[i, k]]):
                    has_nan = True
                    break

            if has_nan:
                result[s, i] = np.nan
            else:
                result[s, i] = val_sum

    return result
