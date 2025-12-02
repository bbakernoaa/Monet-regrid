"""
Numba-optimized polygon clipping algorithms for conservative regridding.

This module implements the Sutherland-Hodgman algorithm for clipping polygons
and calculating intersection areas, designed for 2D curvilinear grids.
"""

import numpy as np
from numba import jit, prange

@jit(nopython=True, nogil=True)
def polygon_area(vertices):
    """
    Calculate the area of a polygon using the shoelace formula.

    Args:
        vertices: (N, 2) array of (x, y) coordinates.

    Returns:
        float: Area of the polygon.
    """
    n = vertices.shape[0]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return 0.5 * abs(area)

@jit(nopython=True, nogil=True)
def is_inside(p1, p2, q):
    """
    Check if point q is inside the edge defined by p1 -> p2.
    (Assuming counter-clockwise ordering, 'inside' is to the left).
    """
    # Cross product (p2-p1) x (q-p1)
    return (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]) >= 0

@jit(nopython=True, nogil=True)
def intersection(p1, p2, p3, p4):
    """
    Find intersection point of line p1->p2 and p3->p4.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Parallel lines

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return np.array([x, y])

@jit(nopython=True, nogil=True)
def clip_polygon(subject_polygon, clip_polygon):
    """
    Clip subject_polygon against clip_polygon using Sutherland-Hodgman algorithm.

    Args:
        subject_polygon: (N, 2) array of vertices
        clip_polygon: (M, 2) array of vertices (must be convex)

    Returns:
        (K, 2) array of vertices of the intersection polygon
    """
    output_list = subject_polygon.copy()

    # Iterate over each edge of the clip polygon
    for i in range(clip_polygon.shape[0]):
        # Define the clip edge
        c1 = clip_polygon[i]
        c2 = clip_polygon[(i + 1) % clip_polygon.shape[0]]

        input_list = output_list
        output_list_len = 0

        # We need a temporary buffer because we can't dynamic append in nopython mode easily
        # Max vertices usually < N+M. Let's preallocate safe buffer.
        # For grids, cells are quads (4), intersection usually max 8-10 vertices.
        temp_output = np.zeros((20, 2), dtype=subject_polygon.dtype)

        if len(input_list) == 0:
            break

        s = input_list[-1]

        for j in range(len(input_list)):
            e = input_list[j]

            if is_inside(c1, c2, e):
                if not is_inside(c1, c2, s):
                    inter = intersection(c1, c2, s, e)
                    if inter is not None:
                        temp_output[output_list_len] = inter
                        output_list_len += 1
                temp_output[output_list_len] = e
                output_list_len += 1
            elif is_inside(c1, c2, s):
                inter = intersection(c1, c2, s, e)
                if inter is not None:
                    temp_output[output_list_len] = inter
                    output_list_len += 1

            s = e

        output_list = temp_output[:output_list_len].copy()

    return output_list

@jit(nopython=True, nogil=True)
def calculate_overlap_area(source_cell, target_cell):
    """
    Calculate the intersection area between two quadrilateral cells.

    Args:
        source_cell: (4, 2) vertices
        target_cell: (4, 2) vertices

    Returns:
        float: Intersection area
    """
    clipped_poly = clip_polygon(source_cell, target_cell)
    if len(clipped_poly) < 3:
        return 0.0
    return polygon_area(clipped_poly)

@jit(nopython=True, nogil=True, parallel=True)
def compute_conservative_weights(
    source_vertices,  # (n_source, 4, 2)
    target_vertices,  # (n_target, 4, 2)
    candidate_indices, # List of list-like or padded array (n_target, max_candidates)
    candidate_counts, # (n_target,)
):
    """
    Compute conservative regridding weights.

    Args:
        source_vertices: Array of source cell vertices
        target_vertices: Array of target cell vertices
        candidate_indices: Indices of potential source cells for each target cell
        candidate_counts: Number of candidates for each target cell

    Returns:
        Tuple of (indices, weights, counts) for sparse matrix construction
        We return flattened arrays: (n_total_interactions, )
    """
    n_targets = target_vertices.shape[0]

    # First pass: Count valid intersections to allocate memory
    # This is hard in parallel without atomics or pre-allocation.
    # We'll assume a max density or do two passes.
    # For now, let's return a dense-ish structure or padded.

    # Actually, constructing sparse matrix is easier if we return
    # arrays of (target_idx, source_idx, weight)

    # Let's use a conservative upper bound for allocation
    # Assume max 16 overlaps per target cell (usually 4-9)
    max_overlaps = 16
    n_total = n_targets * max_overlaps

    out_target_indices = np.full(n_total, -1, dtype=np.int32)
    out_source_indices = np.full(n_total, -1, dtype=np.int32)
    out_weights = np.zeros(n_total, dtype=np.float64)

    # We can't easily parallelize the writing to a single array without knowing offsets.
    # Strategy: Parallelize over targets, write to pre-allocated chunks?
    # Or just use a simple loop if Numba parallel reduction is hard.

    # Return 1D flattened arrays with counts to save memory
    # We first compute counts in a parallel loop, then allocate, then fill

    # Pass 1: Count overlaps per target
    overlap_counts = np.zeros(n_targets, dtype=np.int32)

    for t_idx in prange(n_targets):
        t_poly = target_vertices[t_idx]
        t_area = polygon_area(t_poly)

        if t_area < 1e-12:
            continue

        n_candidates = candidate_counts[t_idx]

        for k in range(n_candidates):
            s_idx = candidate_indices[t_idx, k]
            if s_idx == -1:
                break

            s_poly = source_vertices[s_idx]
            overlap = calculate_overlap_area(s_poly, t_poly)

            if overlap > 1e-12:
                overlap_counts[t_idx] += 1

    # Compute offsets for flattened arrays
    offsets = np.zeros(n_targets + 1, dtype=np.int32)
    # Numba doesn't support cumsum well on arrays in nopython mode sometimes, but let's try manual loop or objmode
    # For parallel safety we need prefix sum. Sequential prefix sum is fast enough for 1D array.

    total_overlaps = 0
    for i in range(n_targets):
        offsets[i] = total_overlaps
        total_overlaps += overlap_counts[i]
    offsets[n_targets] = total_overlaps

    # Allocate flattened arrays
    out_source_indices = np.full(total_overlaps, -1, dtype=np.int32)
    out_weights = np.zeros(total_overlaps, dtype=np.float64)
    out_target_indices = np.zeros(total_overlaps, dtype=np.int32)

    # Pass 2: Fill arrays
    for t_idx in prange(n_targets):
        count = overlap_counts[t_idx]
        if count == 0:
            continue

        start_idx = offsets[t_idx]
        current_idx = start_idx

        t_poly = target_vertices[t_idx]
        t_area = polygon_area(t_poly)

        n_candidates = candidate_counts[t_idx]

        for k in range(n_candidates):
            s_idx = candidate_indices[t_idx, k]
            if s_idx == -1:
                break

            s_poly = source_vertices[s_idx]
            overlap = calculate_overlap_area(s_poly, t_poly)

            if overlap > 1e-12:
                weight = overlap / t_area
                out_source_indices[current_idx] = s_idx
                out_weights[current_idx] = weight
                out_target_indices[current_idx] = t_idx
                current_idx += 1

    return out_source_indices, out_weights, out_target_indices
