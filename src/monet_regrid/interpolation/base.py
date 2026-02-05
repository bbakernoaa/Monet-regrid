"""
Base classes and types for interpolation.

This module provides common utilities, adapters, and availability checks for
various interpolation backends, including Numba-accelerated kernels and
optimized KDTree implementations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

# Try to use pykdtree for faster KDTree operations if available
try:
    from pykdtree.kdtree import KDTree as PyKDTree

    HAS_PYKDTREE = True
except ImportError:
    HAS_PYKDTREE = False

if HAS_PYKDTREE:

    class cKDTree:  # noqa: N801
        """Adapter for pykdtree to mimic scipy.spatial.cKDTree.

        This class provides a subset of the scipy.spatial.cKDTree interface,
        powered by the faster pykdtree library if available.
        """

        def __init__(self, data: np.ndarray, leafsize: int = 10) -> None:
            """Initialize the KDTree.

            Parameters
            ----------
            data : np.ndarray
                The data points to index (n, m).
            leafsize : int, default: 10
                The number of points at which to switch to brute-force.
            """
            self._tree = PyKDTree(data, leafsize=leafsize)
            self._data = data
            self._leafsize = leafsize
            self.n = len(data)

        def __getstate__(self) -> tuple[np.ndarray, int]:
            """Prepare the KDTree for pickling.

            Returns
            -------
            tuple[np.ndarray, int]
                The state needed to reconstruct the KDTree.
            """
            # Pickling support: pykdtree might not pickle well, so we rebuild
            return (self._data, self._leafsize)

        def __setstate__(self, state: tuple[np.ndarray, int]) -> None:
            """Restore the KDTree from a pickled state.

            Parameters
            ----------
            state : tuple[np.ndarray, int]
                The state needed to reconstruct the KDTree.
            """
            data, leafsize = state
            self._data = data
            self._leafsize = leafsize
            self.n = len(data)

        @property
        def data(self) -> np.ndarray:
            """The indexed data points.

            Returns
            -------
            np.ndarray
                The data points.
            """
            return self._data

        def query(
            self,
            x: np.ndarray,
            k: int = 1,
            distance_upper_bound: float = np.inf,
            workers: int = 1,  # noqa: ARG002
        ) -> tuple[np.ndarray | float, np.ndarray | int]:
            """Query the KDTree for nearest neighbors.

            Parameters
            ----------
            x : np.ndarray
                The point or points to query.
            k : int, default: 1
                The number of nearest neighbors to return.
            distance_upper_bound : float, default: inf
                Return only neighbors within this distance.
            workers : int, default: 1
                Number of workers for the query. Ignored by pykdtree but
                kept for API compatibility.

            Returns
            -------
            tuple[np.ndarray | float, np.ndarray | int]
                The distances and indices of the nearest neighbors.
            """
            x = np.asarray(x)
            is_1d = x.ndim == 1
            if is_1d:
                x = x.reshape(1, -1)

            # pykdtree returns (dist, idx)
            d, i = self._tree.query(x, k=k, distance_upper_bound=distance_upper_bound)

            if is_1d:
                if k == 1:
                    return float(d[0]), int(i[0])
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
        compute_linear_weights_grid,
        compute_structured_weights,
    )

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    apply_weights_conservative: Callable[..., Any] | None = None
    apply_weights_linear: Callable[..., Any] | None = None
    apply_weights_nearest: Callable[..., Any] | None = None
    apply_weights_structured: Callable[..., Any] | None = None
    compute_linear_weights_grid: Callable[..., Any] | None = None
    compute_structured_weights: Callable[..., Any] | None = None
    warnings.warn("Numba not available. Falling back to slower pure Python/NumPy implementation.", stacklevel=2)

try:
    from monet_regrid.methods._polygon_clipping import compute_conservative_weights

    HAS_POLYGON_CLIPPING = True
except ImportError:
    HAS_POLYGON_CLIPPING = False
    compute_conservative_weights: Callable[..., Any] | None = None

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
    "compute_linear_weights_grid",
    "compute_structured_weights",
]
