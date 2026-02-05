"""
Numba-optimized polygon clipping algorithms for conservative regridding.

This module implements the Sutherland-Hodgman algorithm for clipping polygons
and calculating intersection areas, designed for 2D curvilinear grids.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True)
def polygon_area(vertices: np.ndarray) -> float:
    """Calculate the area of a polygon using the shoelace formula.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 2) array of (x, y) coordinates.

    Returns
    -------
    float
        Area of the polygon.

    Examples
    --------
    >>> import numpy as np
    >>> vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
    >>> polygon_area(vertices)
    1.0
    """
    n = vertices.shape[0]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return 0.5 * abs(area)


@jit(nopython=True, nogil=True)
def is_inside(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> bool:
    """Check if point q is inside the edge defined by p1 -> p2.

    (Assuming counter-clockwise ordering, 'inside' is to the left).

    Parameters
    ----------
    p1 : np.ndarray
        Start point of the edge (2,).
    p2 : np.ndarray
        End point of the edge (2,).
    q : np.ndarray
        Point to check (2,).

    Returns
    -------
    bool
        True if q is to the left of p1->p2, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> p1, p2 = np.array([0, 0]), np.array([1, 0])
    >>> q = np.array([0.5, 0.5])
    >>> is_inside(p1, p2, q)
    True
    """
    # Cross product (p2-p1) x (q-p1)
    return (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]) >= 0


@jit(nopython=True, nogil=True)
def intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray | None:
    """Find intersection point of line p1->p2 and p3->p4.

    Parameters
    ----------
    p1, p2 : np.ndarray
        Points defining the first line (2,).
    p3, p4 : np.ndarray
        Points defining the second line (2,).

    Returns
    -------
    np.ndarray | None
        Intersection point (2,) or None if lines are parallel.

    Examples
    --------
    >>> import numpy as np
    >>> p1, p2 = np.array([0, 0]), np.array([1, 1])
    >>> p3, p4 = np.array([0, 1]), np.array([1, 0])
    >>> intersection(p1, p2, p3, p4)
    array([0.5, 0.5])
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
def clip_polygon(
    subject_polygon: np.ndarray,
    clip_polygon: np.ndarray,
    temp_buffer1: np.ndarray,
    temp_buffer2: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Clip subject_polygon against clip_polygon using Sutherland-Hodgman algorithm.

    Uses pre-allocated buffers to avoid repeated allocations.

    Parameters
    ----------
    subject_polygon : np.ndarray
        (N, 2) array of vertices.
    clip_polygon : np.ndarray
        (M, 2) array of vertices (must be convex).
    temp_buffer1 : np.ndarray
        (max_v, 2) temporary buffer.
    temp_buffer2 : np.ndarray
        (max_v, 2) temporary buffer.

    Returns
    -------
    tuple[np.ndarray, int]
        (buffer, length) of the intersection polygon vertices.

    Examples
    --------
    >>> import numpy as np
    >>> sub = np.array([[0,0], [2,0], [2,2], [0,2]], dtype=np.float64)
    >>> clip = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float64)
    >>> b1 = np.zeros((20, 2))
    >>> b2 = np.zeros((20, 2))
    >>> res, length = clip_polygon(sub, clip, b1, b2)
    >>> length
    4
    """
    # Initialize output list from subject_polygon
    n_subj = subject_polygon.shape[0]
    for i in range(n_subj):
        temp_buffer1[i] = subject_polygon[i]
    output_len = n_subj
    current_out = temp_buffer1
    current_in = temp_buffer2

    # Iterate over each edge of the clip polygon
    for i in range(clip_polygon.shape[0]):
        # Swap buffers
        if i % 2 == 0:
            current_in = temp_buffer1
            current_out = temp_buffer2
        else:
            current_in = temp_buffer2
            current_out = temp_buffer1

        input_len = output_len
        if input_len == 0:
            return current_out, 0

        # Define the clip edge
        c1 = clip_polygon[i]
        c2 = clip_polygon[(i + 1) % clip_polygon.shape[0]]

        output_len = 0
        s = current_in[input_len - 1]

        for j in range(input_len):
            e = current_in[j]

            if is_inside(c1, c2, e):
                if not is_inside(c1, c2, s):
                    inter = intersection(c1, c2, s, e)
                    if inter is not None:
                        current_out[output_len] = inter
                        output_len += 1
                current_out[output_len] = e
                output_len += 1
            elif is_inside(c1, c2, s):
                inter = intersection(c1, c2, s, e)
                if inter is not None:
                    current_out[output_len] = inter
                    output_len += 1

            s = e

    return current_out, output_len


@jit(nopython=True, nogil=True)
def calculate_overlap_area(
    source_cell: np.ndarray,
    target_cell: np.ndarray,
    buffer1: np.ndarray,
    buffer2: np.ndarray,
) -> float:
    """Calculate the intersection area between two quadrilateral cells.

    Parameters
    ----------
    source_cell : np.ndarray
        (4, 2) vertices.
    target_cell : np.ndarray
        (4, 2) vertices.
    buffer1 : np.ndarray
        (max_v, 2) temporary buffer.
    buffer2 : np.ndarray
        (max_v, 2) temporary buffer.

    Returns
    -------
    float
        Intersection area.

    Examples
    --------
    >>> import numpy as np
    >>> src = np.array([[0,0], [2,0], [2,2], [0,2]], dtype=np.float64)
    >>> tgt = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float64)
    >>> b1, b2 = np.zeros((20, 2)), np.zeros((20, 2))
    >>> calculate_overlap_area(src, tgt, b1, b2)
    1.0
    """
    clipped_poly, length = clip_polygon(source_cell, target_cell, buffer1, buffer2)
    if length < 3:
        return 0.0
    return polygon_area(clipped_poly[:length])


@jit(nopython=True, nogil=True, parallel=True)
def compute_conservative_weights(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute conservative regridding weights.

    Parameters
    ----------
    source_vertices : np.ndarray
        Array of source cell vertices (n_source, 4, 2).
    target_vertices : np.ndarray
        Array of target cell vertices (n_target, 4, 2).
    candidate_indices : np.ndarray
        Indices of potential source cells for each target cell (n_target, max_candidates).
    candidate_counts : np.ndarray
        Number of candidates for each target cell (n_target,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Flattened arrays of (source_indices, weights, target_indices).

    Examples
    --------
    >>> import numpy as np
    >>> src = np.array([[[0,0], [2,0], [2,2], [0,2]]], dtype=np.float64)
    >>> tgt = np.array([[[0,0], [1,0], [1,1], [0,1]]], dtype=np.float64)
    >>> cand = np.array([[0]], dtype=np.int32)
    >>> cnt = np.array([1], dtype=np.int32)
    >>> compute_conservative_weights(src, tgt, cand, cnt)
    (array([0], dtype=int32), array([1.]), array([0], dtype=int32))
    """
    n_targets = target_vertices.shape[0]

    # Max candidates per target to use for caching weights to avoid second pass
    # of expensive clipping algorithm.
    max_candidates_total = candidate_indices.shape[1]

    # Cache for weights found in first pass
    # (n_targets, max_candidates)
    cached_weights = np.zeros((n_targets, max_candidates_total), dtype=np.float64)
    overlap_counts = np.zeros(n_targets, dtype=np.int32)

    # Pass 1: Compute weights and count overlaps per target
    for t_idx in prange(n_targets):
        # Thread-local buffers
        buf1 = np.zeros((20, 2), dtype=source_vertices.dtype)
        buf2 = np.zeros((20, 2), dtype=source_vertices.dtype)

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
            overlap = calculate_overlap_area(s_poly, t_poly, buf1, buf2)

            if overlap > 1e-12:
                weight = overlap / t_area
                cached_weights[t_idx, k] = weight
                overlap_counts[t_idx] += 1

    # Compute offsets for flattened arrays
    offsets = np.zeros(n_targets + 1, dtype=np.int32)
    total_overlaps = 0
    for i in range(n_targets):
        offsets[i] = total_overlaps
        total_overlaps += overlap_counts[i]
    offsets[n_targets] = total_overlaps

    # Allocate flattened arrays
    out_source_indices = np.full(total_overlaps, -1, dtype=np.int32)
    out_weights = np.zeros(total_overlaps, dtype=np.float64)
    out_target_indices = np.zeros(total_overlaps, dtype=np.int32)

    # Pass 2: Fill arrays from cache
    for t_idx in prange(n_targets):
        count = overlap_counts[t_idx]
        if count == 0:
            continue

        start_idx = offsets[t_idx]
        current_idx = start_idx
        n_candidates = candidate_counts[t_idx]

        for k in range(n_candidates):
            weight = cached_weights[t_idx, k]
            if weight > 0:
                s_idx = candidate_indices[t_idx, k]
                out_source_indices[current_idx] = s_idx
                out_weights[current_idx] = weight
                out_target_indices[current_idx] = t_idx
                current_idx += 1
                if current_idx - start_idx == count:
                    break

    return out_source_indices, out_weights, out_target_indices
