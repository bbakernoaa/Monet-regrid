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
