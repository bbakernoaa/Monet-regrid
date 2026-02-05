import numpy as np

from monet_regrid.interpolation import InterpolationEngine


def test_conservative_fallback_vectorization():
    """Test that the conservative regridding fallback produces correct results."""
    # Create a simple interpolation problem
    n_source = 10
    n_targets = 5
    n_samples = 2

    # Mock data: (n_samples, n_source)
    source_data = np.arange(n_samples * n_source).reshape(n_samples, n_source).astype(float)

    # Mock weights: each target gets a mix of two source points
    source_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    target_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    weights = np.array([0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.2, 0.8, 0.9, 0.1])

    engine = InterpolationEngine(method="conservative")
    engine.precomputed_weights = {
        "source_indices": source_indices,
        "target_indices": target_indices,
        "weights": weights,
        "n_targets": n_targets,
        "type": "conservative",
    }

    # Force fallback by temporarily disabling Numba flag if it was enabled
    from monet_regrid.interpolation import core

    original_has_numba = core.HAS_NUMBA
    core.HAS_NUMBA = False

    try:
        result = engine.interpolate(source_data)

        # Expected results
        # target 0: 0.6*src[0] + 0.4*src[1]
        assert np.allclose(result[0, 0], 0.4)
        assert np.allclose(result[1, 0], 10.4)

        # target 2: 0.5*src[4] + 0.5*src[5]
        assert np.allclose(result[0, 2], 4.5)
        assert np.allclose(result[1, 2], 14.5)

    finally:
        core.HAS_NUMBA = original_has_numba


def test_nearest_fallback_vectorization():
    """Test that the nearest neighbor fallback produces correct results."""
    n_source = 10
    n_samples = 2

    source_data = np.arange(n_samples * n_source).reshape(n_samples, n_source).astype(float)

    engine = InterpolationEngine(method="nearest")
    engine.source_indices = np.array([0, 5, 9], dtype=np.int32)
    engine.distances = np.array([0.1, 0.2, 0.3])
    engine.distance_threshold = 0.5

    from monet_regrid.interpolation import core

    original_has_numba = core.HAS_NUMBA
    core.HAS_NUMBA = False

    try:
        result = engine.interpolate(source_data)

        # target 0 -> source 0
        assert np.allclose(result[:, 0], source_data[:, 0])
        # target 1 -> source 5
        assert np.allclose(result[:, 1], source_data[:, 5])
        # target 2 -> source 9
        assert np.allclose(result[:, 2], source_data[:, 9])

    finally:
        core.HAS_NUMBA = original_has_numba


def test_linear_fallback_vectorization():
    """Test that the linear interpolation fallback produces correct results."""
    n_source = 10
    n_samples = 1

    source_data = np.arange(n_samples * n_source).reshape(n_samples, n_source).astype(float)

    engine = InterpolationEngine(method="linear")
    # Simulating 3D linear (4 vertices)
    engine.precomputed_weights = {
        "simplex_indices": np.array([0, -2], dtype=np.int32),
        "barycentric_weights": np.array([[0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0]]),
        "valid_points": np.array([True, True]),
        "fallback_indices": np.array([-1, 9], dtype=np.int32),
        "type": "linear",
    }
    # Mock simplex vertices for simplex 0
    engine._simplex_vertices_cache = np.array([[0, 1, 2, 3]], dtype=np.int32)

    from monet_regrid.interpolation import core

    original_has_numba = core.HAS_NUMBA
    core.HAS_NUMBA = False

    try:
        result = engine.interpolate(source_data)

        # target 0: 0.25*(0+1+2+3) = 1.5
        assert np.allclose(result[0, 0], 1.5)
        # target 1: fallback to source 9
        assert np.allclose(result[0, 1], source_data[0, 9])

    finally:
        core.HAS_NUMBA = original_has_numba


def test_structured_fallback_vectorization():
    """Test that the structured (bilinear) interpolation fallback produces correct results."""
    n_source = 16
    n_samples = 1

    source_data = np.arange(n_samples * n_source).reshape(n_samples, n_source).astype(float)

    engine = InterpolationEngine(method="bilinear")
    # Simulating 4 neighbors per target (bilinear)
    engine.precomputed_weights = {
        "indices": np.array([[0, 1, 4, 5], [10, 11, 14, 15]], dtype=np.int32),
        "weights": np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]),
        "valid_mask": np.array([True, True]),
        "type": "bilinear",
    }

    from monet_regrid.interpolation import core

    original_has_numba = core.HAS_NUMBA
    core.HAS_NUMBA = False

    try:
        result = engine.interpolate(source_data)

        # target 0: 0.25*(src[0]+src[1]+src[4]+src[5]) = 0.25*(0+1+4+5) = 2.5
        assert np.allclose(result[0, 0], 2.5)
        # target 1: 0.1*10 + 0.2*11 + 0.3*14 + 0.4*15 = 1.0 + 2.2 + 4.2 + 6.0 = 13.4
        assert np.allclose(result[0, 1], 13.4)

    finally:
        core.HAS_NUMBA = original_has_numba
