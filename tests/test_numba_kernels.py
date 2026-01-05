"""
Tests for the Numba-optimized kernels.
"""

import numpy as np

from monet_regrid.methods import _numba_kernels


def test_apply_weights_linear():
    """Test the apply_weights_linear kernel."""
    data_flat = np.arange(10, dtype=np.float64).reshape(1, 10)
    simplex_indices = np.array([0, 0, 1], dtype=np.int32)
    barycentric_weights = np.array(
        [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0, 0], [1, 0, 0, 0]],
        dtype=np.float64,
    )
    valid_points = np.array([True, True, True], dtype=np.bool_)
    simplex_vertices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    fallback_indices = np.array([-1, -1, -1], dtype=np.int32)

    result = _numba_kernels.apply_weights_linear(
        data_flat,
        simplex_indices,
        barycentric_weights,
        valid_points,
        simplex_vertices,
        fallback_indices,
    )

    expected = np.array(
        [
            [
                0.25 * 0 + 0.25 * 1 + 0.25 * 2 + 0.25 * 3,
                0.5 * 0 + 0.5 * 1 + 0 * 2 + 0 * 3,
                1 * 4 + 0 * 5 + 0 * 6 + 0 * 7,
            ]
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_apply_weights_nearest():
    """Test the apply_weights_nearest kernel."""
    data_flat = np.arange(10, dtype=np.float64).reshape(1, 10)
    source_indices = np.array([5, 2, 8], dtype=np.int32)
    valid_points = np.array([True, True, True], dtype=np.bool_)

    result = _numba_kernels.apply_weights_nearest(
        data_flat,
        source_indices,
        valid_points,
    )

    expected = np.array([[5.0, 2.0, 8.0]])
    np.testing.assert_allclose(result, expected)


def test_apply_weights_conservative():
    """Test the apply_weights_conservative kernel."""
    data_flat = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    source_indices = np.array([0, 1, 1, 2, 3, 3], dtype=np.int32)
    target_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    weights = np.array([0.5, 0.5, 0.2, 0.8, 0.1, 0.9], dtype=np.float64)
    n_targets = 3

    result = _numba_kernels.apply_weights_conservative(
        data_flat,
        source_indices,
        target_indices,
        weights,
        n_targets,
    )

    expected = np.array([[1 * 0.5 + 2 * 0.5, 2 * 0.2 + 3 * 0.8, 4 * 0.1 + 4 * 0.9]])
    np.testing.assert_allclose(result, expected)


def test_inverse_bilinear_interpolation():
    """Test the inverse_bilinear_interpolation kernel."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([1.0, 1.0])
    v4 = np.array([0.0, 1.0])

    p = np.array([0.5, 0.5])
    u, v = _numba_kernels.inverse_bilinear_interpolation(p, v1, v2, v3, v4)
    np.testing.assert_allclose((u, v), (0.5, 0.5))

    p = np.array([0.25, 0.75])
    u, v = _numba_kernels.inverse_bilinear_interpolation(p, v1, v2, v3, v4)
    np.testing.assert_allclose((u, v), (0.25, 0.75), atol=1e-5)


def test_compute_structured_weights_bilinear():
    """Test the compute_structured_weights kernel with bilinear interpolation."""
    target_points = np.array([[0.5, 0.5]], dtype=np.float64)
    source_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    nearest_indices = np.array([0], dtype=np.int32)
    source_shape = (2, 2)
    method_enum = 0  # Bilinear

    indices, weights, valid = _numba_kernels.compute_structured_weights(
        target_points,
        source_points,
        nearest_indices,
        source_shape,
        method_enum,
    )

    assert valid[0]
    np.testing.assert_allclose(indices[0, :4], np.array([0, 1, 3, 2], dtype=np.int32))
    np.testing.assert_allclose(weights[0, :4], np.array([0.25, 0.25, 0.25, 0.25]))


def test_apply_weights_structured():
    """Test the apply_weights_structured kernel."""
    data_flat = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    weights = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    valid_mask = np.array([True], dtype=np.bool_)

    result = _numba_kernels.apply_weights_structured(
        data_flat,
        indices,
        weights,
        valid_mask,
    )

    expected = np.array([[1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4]])
    np.testing.assert_allclose(result, expected)
