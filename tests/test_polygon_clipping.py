"""
Tests for the Numba-optimized polygon clipping algorithms.
"""

import numpy as np
import pytest

from monet_regrid.methods import _polygon_clipping


def test_polygon_area():
    """Test the polygon_area function."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = _polygon_clipping.polygon_area(vertices)
    np.testing.assert_allclose(area, 1.0)


def test_is_inside():
    """Test the is_inside function."""
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    q_inside = np.array([0.5, -0.1])  # Right-hand rule, y-axis is flipped
    q_outside = np.array([0.5, 0.1])
    assert not _polygon_clipping.is_inside(p1, p2, q_inside)
    assert _polygon_clipping.is_inside(p1, p2, q_outside)


def test_intersection():
    """Test the intersection function."""
    p1 = np.array([0, 0])
    p2 = np.array([1, 1])
    p3 = np.array([0, 1])
    p4 = np.array([1, 0])
    intersection_point = _polygon_clipping.intersection(p1, p2, p3, p4)
    np.testing.assert_allclose(intersection_point, np.array([0.5, 0.5]))


def test_clip_polygon():
    """Test the clip_polygon function."""
    subject_polygon = np.array([[0.5, -0.5], [1.5, 0.5], [0.5, 1.5], [-0.5, 0.5]], dtype=np.float64)
    clip_polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    clipped_polygon = _polygon_clipping.clip_polygon(subject_polygon, clip_polygon)
    area = _polygon_clipping.polygon_area(clipped_polygon)
    np.testing.assert_allclose(area, 1.0, atol=1e-7)


def test_calculate_overlap_area():
    """Test the calculate_overlap_area function."""
    source_cell = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    target_cell = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], dtype=np.float64)
    overlap_area = _polygon_clipping.calculate_overlap_area(source_cell, target_cell)
    np.testing.assert_allclose(overlap_area, 0.25)


def test_compute_conservative_weights():
    """Test the compute_conservative_weights function."""
    source_vertices = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float64)
    target_vertices = np.array(
        [[[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]],
        dtype=np.float64,
    )
    candidate_indices = np.array([[0]], dtype=np.int32)
    candidate_counts = np.array([1], dtype=np.int32)

    (
        source_indices,
        weights,
        target_indices,
    ) = _polygon_clipping.compute_conservative_weights(
        source_vertices,
        target_vertices,
        candidate_indices,
        candidate_counts,
    )

    np.testing.assert_allclose(source_indices, np.array([0]))
    np.testing.assert_allclose(weights, np.array([0.25]))
    np.testing.assert_allclose(target_indices, np.array([0]))
