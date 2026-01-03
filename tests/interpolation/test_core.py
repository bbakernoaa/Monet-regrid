"""
Tests for the interpolation engine.
"""

import numpy as np
import pytest

from monet_regrid.interpolation import core


def test_interpolation_engine_unsupported_method():
    """Test that an unsupported method raises a ValueError."""
    with pytest.raises(ValueError):
        engine = core.InterpolationEngine(method="invalid_method")
        engine.build_structures(np.zeros((2, 3)), np.zeros((2, 3)))


def test_build_structures_conservative_error():
    """Test that build_structures raises an error for conservative method."""
    engine = core.InterpolationEngine(method="conservative")
    with pytest.raises(ValueError):
        engine.build_structures(np.zeros((2, 3)), np.zeros((2, 3)))


def test_build_structures_structured_error():
    """Test that build_structures raises an error for structured method without shape."""
    engine = core.InterpolationEngine(method="bilinear")
    with pytest.raises(ValueError):
        engine.build_structures(np.zeros((2, 3)), np.zeros((2, 3)))


def test_interpolate_unsupported_method():
    """Test that interpolate raises an error for an unsupported method."""
    engine = core.InterpolationEngine(method="nearest")
    engine.method = "invalid_method"
    with pytest.raises(ValueError):
        engine.interpolate(np.zeros((2, 2)))


def test_interpolate_conservative_no_precomputed_weights():
    """Test that conservative interpolation raises an error without precomputed weights."""
    engine = core.InterpolationEngine(method="conservative")
    with pytest.raises(RuntimeError):
        engine.interpolate(np.zeros((2, 2)))


def test_interpolate_structured_no_precomputed_weights():
    """Test that structured interpolation raises an error without precomputed weights."""
    engine = core.InterpolationEngine(method="bilinear")
    with pytest.raises(RuntimeError):
        engine.interpolate(np.zeros((2, 2)))


def test_linear_interpolation_fallback_to_nearest():
    """Test that linear interpolation falls back to nearest neighbor with too few points."""
    engine = core.InterpolationEngine(method="linear")
    source_points = np.random.rand(3, 3)
    target_points = np.random.rand(2, 3)
    engine.build_structures(source_points, target_points)
    assert engine.method == "nearest"


def test_linear_interpolation_direct_computation_error():
    """Test that direct linear interpolation raises a NotImplementedError."""
    engine = core.InterpolationEngine(method="linear")
    with pytest.raises(NotImplementedError):
        engine.interpolate(np.zeros((2, 2)), use_precomputed=False)
