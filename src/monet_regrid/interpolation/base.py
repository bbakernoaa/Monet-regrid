"""
Base classes and types for interpolation.
"""

from __future__ import annotations

import warnings

import numpy as np

# Try to use pykdtree for faster KDTree operations if available
try:
    from pykdtree.kdtree import KDTree as PyKDTree

    HAS_PYKDTREE = True
except ImportError:
    HAS_PYKDTREE = False

if HAS_PYKDTREE:

    class cKDTree:
        """Adapter for pykdtree to mimic scipy.spatial.cKDTree."""

        def __init__(self, data, leafsize=10):
            self._tree = PyKDTree(data, leafsize=leafsize)
            self._data = data
            self._leafsize = leafsize
            self.n = len(data)

        def __getstate__(self):
            # Pickling support: pykdtree might not pickle well, so we rebuild
            return (self._data, self._leafsize)

        def __setstate__(self, state):
            data, leafsize = state
            self._data = data
            self._leafsize = leafsize
            self.n = len(data)

        @property
        def data(self):
            return self._data

        def query(self, x, k=1, distance_upper_bound=np.inf, workers=1):
            x = np.asarray(x)
            is_1d = x.ndim == 1
            if is_1d:
                x = x.reshape(1, -1)

            # pykdtree returns (dist, idx)
            d, i = self._tree.query(x, k=k, distance_upper_bound=distance_upper_bound)

            if is_1d:
                if k == 1:
                    return d[0], i[0]
                else:
                    return d[0], i[0]
            return d, i
else:
    from scipy.spatial import cKDTree  # type: ignore

try:
    from monet_regrid.methods._numba_kernels import (
        apply_weights_conservative,
        apply_weights_linear,
        apply_weights_nearest,
        apply_weights_structured,
        compute_structured_weights,
    )

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    apply_weights_conservative = None
    apply_weights_linear = None
    apply_weights_nearest = None
    apply_weights_structured = None
    compute_structured_weights = None
    warnings.warn("Numba not available. Falling back to slower pure Python/NumPy implementation.", stacklevel=2)

try:
    from monet_regrid.methods._polygon_clipping import compute_conservative_weights

    HAS_POLYGON_CLIPPING = True
except ImportError:
    HAS_POLYGON_CLIPPING = False
    compute_conservative_weights = None

__all__ = [
    "HAS_NUMBA",
    "HAS_POLYGON_CLIPPING",
    "HAS_PYKDTREE",
    "apply_weights_conservative",
    "apply_weights_linear",
    "apply_weights_nearest",
    "apply_weights_structured",
    "cKDTree",
    "compute_conservative_weights",
    "compute_structured_weights",
]
