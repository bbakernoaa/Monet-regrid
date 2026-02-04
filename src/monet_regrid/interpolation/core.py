"""
Optimized interpolation engine with precomputed weights and vectorized fallbacks.

This module implements the core interpolation logic, supporting multiple methods
(nearest, linear, conservative, bilinear, cubic) with both Numba-accelerated
kernels and optimized vectorized Python/SciPy fallbacks.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay  # type: ignore

from monet_regrid.interpolation.base import (
    HAS_NUMBA,
    HAS_POLYGON_CLIPPING,
    apply_weights_conservative,
    apply_weights_linear,
    apply_weights_nearest,
    apply_weights_structured,
    cKDTree,
    compute_conservative_weights,
    compute_linear_weights_grid,
    compute_structured_weights,
)


class InterpolationEngine:
    """Optimized interpolation engine with precomputed weights.

    Attributes
    ----------
    method : {"nearest", "linear", "conservative", "bilinear", "cubic"}
        Interpolation method to use.
    spherical : bool
        Whether to use spherical barycentrics (True) or planar (False).
    fill_method : {"nan", "nearest"}
        How to handle out-of-domain targets.
    extrapolate : bool
        Whether to allow extrapolation beyond source domain.
    """

    def __init__(
        self,
        method: Literal["nearest", "linear", "conservative", "bilinear", "cubic"] = "linear",
        spherical: bool = True,
        fill_method: Literal["nan", "nearest"] = "nan",
        extrapolate: bool = False,
    ) -> None:
        """Initialize the interpolation engine.

        Parameters
        ----------
        method : {"nearest", "linear", "conservative", "bilinear", "cubic"}, default: "linear"
            Interpolation method to use.
        spherical : bool, default: True
            Whether to use spherical barycentrics (True) or planar (False).
        fill_method : {"nan", "nearest"}, default: "nan"
            How to handle out-of-domain targets.
        extrapolate : bool, default: False
            Whether to allow extrapolation beyond source domain.
        """
        self.method = method
        self.spherical = spherical
        self.fill_method = fill_method
        self.extrapolate = extrapolate

        # Interpolation structures
        self.source_kdtree: cKDTree | None = None
        self.target_kdtree: cKDTree | None = None
        self.triangles: Delaunay | None = None
        self.barycentric_weights: np.ndarray | None = None
        self.source_indices: np.ndarray | None = None
        self.distances: np.ndarray | None = None
        self.distance_threshold: float | None = None

        # Precomputed weights for build-once/apply-many pattern
        self.precomputed_weights: dict[str, Any] | None = None
        self.target_points_3d: np.ndarray | None = None

        # Cache for simple vertices array to avoid object access in loop/kernel
        self._simplex_vertices_cache: np.ndarray | None = None
        self._fallback_indices: np.ndarray | None = None

    def build_structures(
        self,
        source_points_3d: np.ndarray,
        target_points_3d: np.ndarray,
        radius_of_influence: float | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> None:
        """Build interpolation structures based on method.

        Parameters
        ----------
        source_points_3d : np.ndarray
            Array of 3D source points (n, 3).
        target_points_3d : np.ndarray
            Array of 3D target points (m, 3).
        radius_of_influence : float | None, default: None
            Maximum distance for valid interpolation.
        source_shape : tuple[int, int] | None, default: None
            Shape of source grid (ny, nx) for structured interpolation.

        Returns
        -------
        None

        Examples
        --------
        >>> engine = InterpolationEngine(method="nearest")
        >>> src = np.array([[0,0,0], [1,1,1]])
        >>> tgt = np.array([[0.1, 0.1, 0.1]])
        >>> engine.build_structures(src, tgt)
        """
        self.target_points_3d = target_points_3d
        if self.method == "nearest":
            self._build_nearest_neighbour(source_points_3d, target_points_3d, radius_of_influence)
        elif self.method == "linear":
            self._build_linear_interpolation(source_points_3d, target_points_3d, radius_of_influence, source_shape)
        elif self.method in ["bilinear", "cubic"]:
            if source_shape is None:
                msg = f"Method '{self.method}' requires source_shape to be provided."
                raise ValueError(msg)
            self._build_structured_interpolation(source_points_3d, target_points_3d, source_shape, self.method)
        elif self.method == "conservative":
            msg = "Conservative regridding requires grid boundaries. Use build_conservative_structures() instead."
            raise ValueError(msg)
        else:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)

    def build_conservative_structures(
        self,
        source_centers_3d: np.ndarray,
        target_centers_3d: np.ndarray,
        source_vertices_lonlat: np.ndarray,
        target_vertices_lonlat: np.ndarray,
        radius_of_influence: float | None = None,
    ) -> None:
        """Build structures for conservative regridding.

        Parameters
        ----------
        source_centers_3d : np.ndarray
            Centers of source cells (N, 3).
        target_centers_3d : np.ndarray
            Centers of target cells (M, 3).
        source_vertices_lonlat : np.ndarray
            Vertices of source cells (N, 4, 2) in (lon, lat).
        target_vertices_lonlat : np.ndarray
            Vertices of target cells (M, 4, 2) in (lon, lat).
        radius_of_influence : float | None, default: None
            Search radius for overlapping cells.

        Returns
        -------
        None
        """
        if not HAS_POLYGON_CLIPPING:
            msg = "Numba is required for conservative regridding."
            raise ImportError(msg)

        # 1. Build KDTree on source centers to find candidates
        self.source_kdtree = cKDTree(source_centers_3d)

        # 2. Query KDTree to find potential source cells for each target cell
        if radius_of_influence is None:
            radius_of_influence = 500000.0  # 500km default

        # Find candidates
        k_candidates = 25
        _dists, indices = self.source_kdtree.query(target_centers_3d, k=k_candidates, distance_upper_bound=radius_of_influence)

        # indices has shape (M, k). Invalid indices are self.source_kdtree.n
        # We need to clean this up for the kernel
        n_source = source_centers_3d.shape[0]
        indices = np.asarray(indices).astype(np.int32)
        indices[indices == n_source] = -1

        # Create counts
        counts = np.sum(indices != -1, axis=1).astype(np.int32)

        # 3. Compute weights using Numba kernel
        # Numba kernel expects contiguous arrays
        source_vertices_lonlat = np.ascontiguousarray(source_vertices_lonlat)
        target_vertices_lonlat = np.ascontiguousarray(target_vertices_lonlat)

        res_source_indices, res_weights, res_target_indices = compute_conservative_weights(
            source_vertices_lonlat, target_vertices_lonlat, indices, counts
        )

        if len(res_weights) == 0:
            warnings.warn("Conservative regridding found no overlaps. Check coordinates or radius.", stacklevel=2)

        self.precomputed_weights = {
            "source_indices": res_source_indices,
            "target_indices": res_target_indices,
            "weights": res_weights,
            "type": "conservative",
            "n_targets": target_centers_3d.shape[0],
        }

    def _build_structured_interpolation(
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, source_shape: tuple[int, int], method: str
    ) -> None:
        """Build structures for structured interpolation (bilinear/cubic).

        Parameters
        ----------
        source_points_3d : np.ndarray
            Array of 3D source points.
        target_points_3d : np.ndarray
            Array of 3D target points.
        source_shape : tuple[int, int]
            Shape of the source grid (ny, nx).
        method : str
            The interpolation method ("bilinear" or "cubic").
        """
        if not HAS_NUMBA:
            msg = f"Numba is required for {method} regridding."
            raise ImportError(msg)

        # 1. Build KDTree on source points (centers/nodes)
        self.source_kdtree = cKDTree(source_points_3d)

        # 2. Find nearest neighbor for each target point
        _, nearest_indices = self.source_kdtree.query(target_points_3d, k=1)
        nearest_indices = np.asarray(nearest_indices).astype(np.int32)

        # 3. Compute weights using Numba kernel
        method_enum = 0 if method == "bilinear" else 1

        # Correct arguments: target, source
        res_indices, res_weights, valid_mask = compute_structured_weights(
            target_points_3d, source_points_3d, nearest_indices, source_shape, method_enum
        )

        if not np.any(valid_mask):
            warnings.warn(f"{method} interpolation found no valid points. Check geometry.", stacklevel=2)

        self.precomputed_weights = {
            "indices": res_indices,
            "weights": res_weights,
            "valid_mask": valid_mask,
            "type": method,
        }

    def _build_nearest_neighbour(
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, radius_of_influence: float | None = None
    ) -> None:
        """Build KDTree for nearest neighbor interpolation.

        Parameters
        ----------
        source_points_3d : np.ndarray
            Array of 3D source points.
        target_points_3d : np.ndarray
            Array of 3D target points.
        radius_of_influence : float | None, default: None
            Maximum distance for valid interpolation.
        """
        # Create KDTree from source points in 3D space
        self.source_kdtree = cKDTree(source_points_3d)

        # Find nearest source point for each target point
        if self.source_kdtree is not None:
            distances, indices = self.source_kdtree.query(target_points_3d, k=1, workers=-1)

            self.source_indices = np.asarray(indices).astype(np.int32)
            self.distances = np.asarray(distances)

        # Determine a reasonable threshold for identifying out-of-domain points
        if radius_of_influence is not None:
            self.distance_threshold = float(radius_of_influence)
        else:
            self.distance_threshold = float("inf")

    def _build_linear_interpolation(
        self,
        source_points_3d: np.ndarray,
        target_points_3d: np.ndarray,
        radius_of_influence: float | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> None:
        """Build interpolation structures for linear interpolation.

        Parameters
        ----------
        source_points_3d : np.ndarray
            Array of 3D source points.
        target_points_3d : np.ndarray
            Array of 3D target points.
        radius_of_influence : float | None, default: None
            Maximum distance for valid interpolation.
        source_shape : tuple[int, int] | None, default: None
            Shape of source grid for grid-based linear interpolation.
        """
        # If we have a grid structure, use the faster grid-based linear interpolation
        if source_shape is not None and HAS_NUMBA:
            self._build_linear_interpolation_grid(source_points_3d, target_points_3d, source_shape, radius_of_influence)
            return

        # Fallback to Delaunay for unstructured or non-Numba cases
        # Delaunay requires at least N+1 points in N dimensions. For 3D, we need at least 4 points.
        if len(source_points_3d) < 4:
            warnings.warn("Fewer than 4 source points. Falling back to nearest neighbor.", stacklevel=2)
            self.method = "nearest"
            self._build_nearest_neighbour(source_points_3d, target_points_3d, radius_of_influence)
            return

        # Build Delaunay triangulation of source points
        try:
            # Use 'QJ' to joggle input to avoid QhullErrors for coplanar points
            self.triangles = Delaunay(source_points_3d, qhull_options="QJ")
            # Cache vertices array for efficient Numba access
            self._simplex_vertices_cache = self.triangles.simplices.astype(np.int32)
        except Exception as e:
            warnings.warn(
                f"Could not build Delaunay triangulation: {e}. Falling back to nearest neighbor.",
                stacklevel=2,
            )
            self.method = "nearest"
            self._build_nearest_neighbour(source_points_3d, target_points_3d, radius_of_influence)
            return

        # Build KDTree for source points to enable nearest neighbor fallback
        self.source_kdtree = cKDTree(source_points_3d)

        # Set distance_threshold
        if radius_of_influence is not None:
            self.distance_threshold = float(radius_of_influence)

        # Build KDTree for target points to find closest triangles
        self.target_kdtree = cKDTree(target_points_3d)

        # Precompute barycentric coordinates for target points
        self._precompute_barycentric_weights(target_points_3d, source_points_3d)

    def _build_linear_interpolation_grid(
        self,
        source_points_3d: np.ndarray,
        target_points_3d: np.ndarray,
        source_shape: tuple[int, int],
        radius_of_influence: float | None = None,
    ) -> None:
        """Build structures for grid-based linear interpolation.

        Parameters
        ----------
        source_points_3d : np.ndarray
            Array of 3D source points.
        target_points_3d : np.ndarray
            Array of 3D target points.
        source_shape : tuple[int, int]
            Shape of the source grid (ny, nx).
        radius_of_influence : float | None, default: None
            Maximum distance for valid interpolation.
        """
        # 1. Build KDTree on source points
        self.source_kdtree = cKDTree(source_points_3d)

        # 2. Find nearest neighbor for each target point
        _, nearest_indices = self.source_kdtree.query(target_points_3d, k=1, workers=-1)
        nearest_indices = np.asarray(nearest_indices).astype(np.int32)

        # 3. Compute weights using Numba kernel
        res_indices, res_weights, valid_points = compute_linear_weights_grid(
            target_points_3d, source_points_3d, nearest_indices, source_shape
        )

        # 4. Handle points outside the grid or those that failed grid search
        # Fallback to nearest neighbor for those if requested
        if not np.all(valid_points):
            not_found_indices = np.where(~valid_points)[0]
            if self.fill_method == "nearest" or radius_of_influence is not None:
                distances, fallback_idxs = self.source_kdtree.query(target_points_3d[not_found_indices], workers=-1)
                distances = np.asarray(distances)
                fallback_idxs = np.asarray(fallback_idxs)

                for i, idx in enumerate(not_found_indices):
                    if self.fill_method == "nearest" or (radius_of_influence is not None and distances[i] < radius_of_influence):
                        res_indices[idx, 0] = fallback_idxs[i]
                        res_weights[idx, 0] = 1.0
                        res_indices[idx, 1:] = -1
                        res_weights[idx, 1:] = 0.0
                        valid_points[idx] = True

        self.precomputed_weights = {
            "simplex_indices": np.arange(len(target_points_3d), dtype=np.int32),
            "barycentric_weights": res_weights,
            "valid_points": valid_points,
            "fallback_indices": res_indices[:, 0],
            "type": "linear",
        }
        # Adaptation for apply_weights_linear kernel
        self._simplex_vertices_cache = res_indices.astype(np.int32)

    def _precompute_barycentric_weights(self, target_points_3d: np.ndarray, source_points_3d: np.ndarray) -> None:
        """Precompute barycentric weights for all target points.

        Parameters
        ----------
        target_points_3d : np.ndarray
            Array of 3D target points.
        source_points_3d : np.ndarray
            Array of 3D source points.
        """
        n_targets = len(target_points_3d)
        self.precomputed_weights = {
            "simplex_indices": np.full(n_targets, -1, dtype=np.int32),
            "barycentric_weights": np.zeros((n_targets, 4), dtype=np.float64),
            "valid_points": np.zeros(n_targets, dtype=bool),
            "fallback_indices": np.full(n_targets, -1, dtype=np.int32),
            "type": "linear",
        }

        if self.triangles is None:
            msg = "Triangulation not initialized"
            raise RuntimeError(msg)

        if self.fill_method == "nearest" and self.source_kdtree is not None:
            _, fallback_indices = self.source_kdtree.query(target_points_3d)
            self.precomputed_weights["fallback_indices"] = np.asarray(fallback_indices).astype(np.int32)

        # Vectorized find_simplex call
        simplex_indices = self.triangles.find_simplex(target_points_3d, tol=-1e-8)

        # Retry with scaled points for those not found (handles points slightly outside hull)
        not_found_mask = simplex_indices == -1
        points_to_use = target_points_3d.copy()

        if np.any(not_found_mask):
            centroid = np.mean(source_points_3d, axis=0)
            for scale in [0.999, 0.99, 0.95, 0.9]:
                current_not_found_indices = np.where(simplex_indices == -1)[0]
                if len(current_not_found_indices) == 0:
                    break
                points_to_retry = target_points_3d[current_not_found_indices]
                vectors = points_to_retry - centroid
                scaled_points = centroid + vectors * scale
                retry_indices = self.triangles.find_simplex(scaled_points, tol=-1e-8)
                found_in_retry = retry_indices != -1
                if np.any(found_in_retry):
                    original_indices = current_not_found_indices[found_in_retry]
                    simplex_indices[original_indices] = retry_indices[found_in_retry]
                    points_to_use[original_indices] = scaled_points[found_in_retry]

        # Vectorized weight computation
        valid_simplex_mask = simplex_indices != -1
        if np.any(valid_simplex_mask):
            valid_indices = np.where(valid_simplex_mask)[0]
            valid_simplex_indices = simplex_indices[valid_indices]
            vertices_indices = self.triangles.simplices[valid_simplex_indices]
            vertices = source_points_3d[vertices_indices]

            # Construct A matrices and b vectors for batch solve
            a_batch = np.ones((len(valid_indices), 4, 4), dtype=np.float64)
            a_batch[:, :3, :] = vertices.transpose(0, 2, 1)
            b_batch = np.ones((len(valid_indices), 4), dtype=np.float64)
            b_batch[:, :3] = target_points_3d[valid_indices]

            try:
                weights = np.linalg.solve(a_batch, b_batch[..., np.newaxis]).squeeze(-1)
                self.precomputed_weights["simplex_indices"][valid_indices] = valid_simplex_indices
                self.precomputed_weights["barycentric_weights"][valid_indices] = weights
                self.precomputed_weights["valid_points"][valid_indices] = True
            except np.linalg.LinAlgError:
                # Singular matrix fallback (should be rare)
                for i, idx in enumerate(valid_indices):
                    try:
                        w = np.linalg.solve(a_batch[i], b_batch[i])
                        self.precomputed_weights["simplex_indices"][idx] = valid_simplex_indices[i]
                        self.precomputed_weights["barycentric_weights"][idx] = w
                        self.precomputed_weights["valid_points"][idx] = True
                    except np.linalg.LinAlgError:
                        pass

        # Handle points outside the convex hull with distance threshold
        not_found_mask = simplex_indices == -1
        if np.any(not_found_mask):
            not_found_indices = np.where(not_found_mask)[0]
            if self.fill_method == "nearest" and self.source_kdtree is not None:
                self.precomputed_weights["simplex_indices"][not_found_indices] = -2
                self.precomputed_weights["valid_points"][not_found_indices] = True
            elif self.distance_threshold is not None and self.source_kdtree is not None:
                distances, nearest_idxs = self.source_kdtree.query(target_points_3d[not_found_indices], workers=-1)
                distances = np.asarray(distances)
                nearest_idxs = np.asarray(nearest_idxs)
                within_threshold = distances < self.distance_threshold
                if np.any(within_threshold):
                    actual_indices = not_found_indices[within_threshold]
                    self.precomputed_weights["simplex_indices"][actual_indices] = -2
                    self.precomputed_weights["valid_points"][actual_indices] = True
                    self.precomputed_weights["fallback_indices"][actual_indices] = nearest_idxs[within_threshold]

    def interpolate(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Apply interpolation to source data.

        Parameters
        ----------
        source_data : np.ndarray
            Input data array with source grid dimensions.
        use_precomputed : bool, default: True
            Whether to use precomputed weights.

        Returns
        -------
        np.ndarray
            Interpolated data on target grid.

        Examples
        --------
        >>> engine = InterpolationEngine(method="nearest")
        >>> src_points = np.array([[0,0,0], [1,1,1]])
        >>> tgt_points = np.array([[0.1, 0.1, 0.1]])
        >>> engine.build_structures(src_points, tgt_points)
        >>> data = np.array([10.0, 20.0])
        >>> engine.interpolate(data)
        array([10.])
        """
        if self.method == "nearest":
            return self._interpolate_nearest(source_data)
        elif self.method == "linear":
            return self._interpolate_linear(source_data, use_precomputed)
        elif self.method == "conservative":
            return self._interpolate_conservative(source_data, use_precomputed)
        elif self.method in ["bilinear", "cubic"]:
            return self._interpolate_structured(source_data, use_precomputed)
        else:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)

    def _interpolate_conservative(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform conservative regridding using vectorized sparse multiplication.

        Parameters
        ----------
        source_data : np.ndarray
            The input data to regrid.
        use_precomputed : bool, default: True
            Whether to use precomputed weights.

        Returns
        -------
        np.ndarray
            The regridded data.
        """
        if not use_precomputed or self.precomputed_weights is None:
            msg = "Weights not precomputed for conservative regridding."
            raise RuntimeError(msg)

        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        n_other_dims = len(original_shape) - 1
        reshaped_data = source_data.reshape(-1, n_spatial)

        source_indices = self.precomputed_weights["source_indices"]
        target_indices = self.precomputed_weights["target_indices"]
        weights = self.precomputed_weights["weights"]
        max_t_idx = int(target_indices.max()) + 1 if len(target_indices) > 0 else 0
        n_targets = self.precomputed_weights.get("n_targets", max_t_idx)

        if HAS_NUMBA:
            result = apply_weights_conservative(reshaped_data, source_indices, target_indices, weights, n_targets)
        else:
            # Optimized SciPy sparse matrix multiplication fallback
            weight_matrix = sparse.csr_matrix((weights, (target_indices, source_indices)), shape=(n_targets, n_spatial))
            if hasattr(reshaped_data, "tocsr"):
                result = (reshaped_data @ weight_matrix.T).toarray()
            else:
                result = reshaped_data @ weight_matrix.T

        target_shape = (*original_shape[:-1], n_targets) if n_other_dims > 0 else (n_targets,)
        return result.reshape(target_shape)

    def _interpolate_structured(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform structured interpolation (bilinear/cubic) with vectorized fallback.

        Parameters
        ----------
        source_data : np.ndarray
            The input data to interpolate.
        use_precomputed : bool, default: True
            Whether to use precomputed weights.

        Returns
        -------
        np.ndarray
            The interpolated data.
        """
        if not use_precomputed or self.precomputed_weights is None:
            msg = f"Weights not precomputed for {self.method} regridding."
            raise RuntimeError(msg)

        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        reshaped_data = source_data.reshape(-1, n_spatial)

        indices = self.precomputed_weights["indices"]
        weights = self.precomputed_weights["weights"]
        valid_mask = self.precomputed_weights["valid_mask"]
        n_targets = indices.shape[0]

        if HAS_NUMBA:
            result = apply_weights_structured(reshaped_data, indices, weights, valid_mask)
        else:
            # Structured interpolation uses a small fixed number of neighbors per target
            n_neighbors = indices.shape[1]
            result = np.full((reshaped_data.shape[0], n_targets), np.nan, dtype=source_data.dtype)

            # Vectorize the application across non-spatial dimensions
            for i in range(n_neighbors):
                neighbor_indices = indices[:, i]
                neighbor_weights = weights[:, i]
                mask = valid_mask & (neighbor_indices != -1)
                if np.any(mask):
                    contribution = reshaped_data[:, neighbor_indices[mask]] * neighbor_weights[mask]
                    result[:, mask] = np.where(np.isnan(result[:, mask]), 0, result[:, mask]) + contribution

        target_shape = (*original_shape[:-1], n_targets) if len(original_shape) > 1 else (n_targets,)
        return result.reshape(target_shape)

    def _interpolate_nearest(self, source_data: np.ndarray) -> np.ndarray:
        """Perform nearest neighbor interpolation using vectorized indexing.

        Parameters
        ----------
        source_data : np.ndarray
            The input data to interpolate.

        Returns
        -------
        np.ndarray
            The interpolated data.
        """
        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        reshaped_data = source_data.reshape(-1, n_spatial)

        if self.source_indices is None:
            msg = "Source indices not computed"
            raise RuntimeError(msg)

        valid_mask = (
            (self.distances < self.distance_threshold)
            if (self.fill_method == "nan" and self.distances is not None and self.distance_threshold is not None)
            else np.ones(len(self.source_indices), dtype=bool)
        )

        if HAS_NUMBA:
            result = apply_weights_nearest(reshaped_data, self.source_indices, valid_mask)
        else:
            # Fast vectorized NumPy indexing fallback
            result = np.full((reshaped_data.shape[0], len(self.source_indices)), np.nan, dtype=source_data.dtype)
            valid_targets = np.where(valid_mask)[0]
            if len(valid_targets) > 0:
                target_source_indices = self.source_indices[valid_targets]
                result[:, valid_targets] = reshaped_data[:, target_source_indices]

        target_shape = (*original_shape[:-1], len(self.source_indices)) if len(original_shape) > 1 else (len(self.source_indices),)
        return result.reshape(target_shape)

    def _interpolate_linear(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform linear interpolation using vectorized sparse matrix fallback.

        Parameters
        ----------
        source_data : np.ndarray
            The input data to interpolate.
        use_precomputed : bool, default: True
            Whether to use precomputed weights.

        Returns
        -------
        np.ndarray
            The interpolated data.
        """
        if not use_precomputed or self.precomputed_weights is None:
            msg = "Direct linear interpolation computation is not implemented. Use precomputed weights."
            raise NotImplementedError(msg)

        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        reshaped_data = source_data.reshape(-1, n_spatial)
        n_targets = len(self.precomputed_weights["valid_points"])

        if HAS_NUMBA:
            if self._simplex_vertices_cache is None and self.triangles is not None:
                self._simplex_vertices_cache = self.triangles.simplices.astype(np.int32)
            result = apply_weights_linear(
                reshaped_data,
                self.precomputed_weights["simplex_indices"],
                self.precomputed_weights["barycentric_weights"],
                self.precomputed_weights["valid_points"],
                self._simplex_vertices_cache,
                self.precomputed_weights["fallback_indices"],
            )
        else:
            # Optimized SciPy sparse matrix multiplication fallback for linear interpolation
            # Construct weight matrix
            simplex_indices = self.precomputed_weights["simplex_indices"]
            barycentric_weights = self.precomputed_weights["barycentric_weights"]

            valid_mask = simplex_indices >= 0
            fb_mask = simplex_indices == -2

            row_indices = []
            col_indices = []
            weight_values = []

            if np.any(valid_mask):
                valid_target_idxs = np.where(valid_mask)[0]
                valid_simplex_idxs = simplex_indices[valid_mask]
                valid_weights = barycentric_weights[valid_mask]

                # simplices has shape (n_simplices, 4)
                simplices = self._simplex_vertices_cache if self._simplex_vertices_cache is not None else self.triangles.simplices
                vertex_indices = simplices[valid_simplex_idxs]

                row_indices.append(np.repeat(valid_target_idxs, 4))
                col_indices.append(vertex_indices.flatten())
                weight_values.append(valid_weights.flatten())

            if np.any(fb_mask):
                fb_target_idxs = np.where(fb_mask)[0]
                fb_source_idxs = self.precomputed_weights["fallback_indices"][fb_mask]
                row_indices.append(fb_target_idxs)
                col_indices.append(fb_source_idxs)
                weight_values.append(np.ones(len(fb_target_idxs)))

            if row_indices:
                rows = np.concatenate(row_indices)
                cols = np.concatenate(col_indices)
                vals = np.concatenate(weight_values)
                weight_matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n_targets, n_spatial))
                if hasattr(reshaped_data, "tocsr"):
                    result = (reshaped_data @ weight_matrix.T).toarray()
                else:
                    result = reshaped_data @ weight_matrix.T
            else:
                result = np.full((reshaped_data.shape[0], n_targets), np.nan, dtype=source_data.dtype)

        target_shape = (*original_shape[:-1], n_targets) if len(original_shape) > 1 else (n_targets,)
        return result.reshape(target_shape)
