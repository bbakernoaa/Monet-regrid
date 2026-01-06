"""
Optimized interpolation engine with precomputed weights.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
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
    compute_structured_weights,
)
from monet_regrid.interpolation.utils import _compute_barycentric_weights_3d


class InterpolationEngine:
    """Optimized interpolation engine with precomputed weights."""

    def __init__(
        self,
        method: Literal["nearest", "linear", "conservative", "bilinear", "cubic"] = "linear",
        spherical: bool = True,
        fill_method: Literal["nan", "nearest"] = "nan",
        extrapolate: bool = False,
    ):
        """Initialize the interpolation engine.

        Args:
            method: Interpolation method ('nearest', 'linear', 'conservative', 'bilinear', 'cubic')
            spherical: Whether to use spherical barycentrics (True) or planar (False)
            fill_method: How to handle out-of-domain targets ('nan' or 'nearest')
            extrapolate: Whether to allow extrapolation beyond source domain
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

    def build_structures(
        self,
        source_points_3d: np.ndarray,
        target_points_3d: np.ndarray,
        radius_of_influence: float | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> None:
        """Build interpolation structures based on method.

        Args:
            source_points_3d: Array of 3D source points (n, 3)
            target_points_3d: Array of 3D target points (m, 3)
            radius_of_influence: Maximum distance for valid interpolation
            source_shape: Shape of source grid (ny, nx) for structured interpolation
        """
        self.target_points_3d = target_points_3d
        if self.method == "nearest":
            self._build_nearest_neighbour(source_points_3d, target_points_3d, radius_of_influence)
        elif self.method == "linear":
            self._build_linear_interpolation(source_points_3d, target_points_3d, radius_of_influence)
        elif self.method in ["bilinear", "cubic"]:
            if source_shape is None:
                msg = f"Method '{self.method}' requires source_shape to be provided."
                raise ValueError(msg)
            self._build_structured_interpolation(source_points_3d, target_points_3d, source_shape, self.method)
        elif self.method == "conservative":
            # Conservative regridding requires boundaries, so this method shouldn't be called directly
            # with points. It should be called via build_conservative_structures.
            # However, if called, we can raise an error or fallback.
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

        Args:
            source_centers_3d: Centers of source cells (N, 3)
            target_centers_3d: Centers of target cells (M, 3)
            source_vertices_lonlat: Vertices of source cells (N, 4, 2) in (lon, lat)
            target_vertices_lonlat: Vertices of target cells (M, 4, 2) in (lon, lat)
            radius_of_influence: Search radius for overlapping cells
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
        _dists, indices = self.source_kdtree.query(
            target_centers_3d, k=k_candidates, distance_upper_bound=radius_of_influence
        )

        # indices has shape (M, k). Invalid indices are self.source_kdtree.n
        # We need to clean this up for the kernel
        n_source = source_centers_3d.shape[0]
        indices[indices == n_source] = -1

        # Create counts
        counts = np.sum(indices != -1, axis=1).astype(np.int32)
        indices = indices.astype(np.int32)

        # 3. Compute weights using Numba kernel
        # Numba kernel expects contiguous arrays
        source_vertices_lonlat = np.ascontiguousarray(source_vertices_lonlat)
        target_vertices_lonlat = np.ascontiguousarray(target_vertices_lonlat)

        res_source_indices, res_weights, res_target_indices = compute_conservative_weights(
            source_vertices_lonlat, target_vertices_lonlat, indices, counts
        )

        # 4. Store weights in sparse-friendly format
        # The arrays are already flattened and filtered by the Numba kernel (mostly)
        # We might still have some -1s if the allocation was conservative, but our
        # new kernel returns exactly needed size + potentially some padding if we did it that way,
        # but the new kernel implementation uses precise counting, so they should be tight.
        # Actually, the new kernel implementation does 2 passes and returns exact size.

        if len(res_weights) == 0:
            warnings.warn("Conservative regridding found no overlaps. Check coordinates or radius.", stacklevel=2)

        self.precomputed_weights = {
            "source_indices": res_source_indices,
            "target_indices": res_target_indices,
            "weights": res_weights,
            "type": "conservative",
        }

    def _build_structured_interpolation(
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, source_shape: tuple[int, int], method: str
    ) -> None:
        """Build structures for structured interpolation (bilinear/cubic)."""
        if not HAS_NUMBA:
            msg = f"Numba is required for {method} regridding."
            raise ImportError(msg)

        # 1. Build KDTree on source points (centers/nodes)
        self.source_kdtree = cKDTree(source_points_3d)

        # 2. Find nearest neighbor for each target point
        _, nearest_indices = self.source_kdtree.query(target_points_3d, k=1)
        nearest_indices = nearest_indices.astype(np.int32)

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
        """Build KDTree for nearest neighbor interpolation."""
        # Create KDTree from source points in 3D space
        self.source_kdtree = cKDTree(source_points_3d)

        # For nearest neighbor, we just need to query the tree
        # Find nearest source point for each target point
        if self.source_kdtree is not None:
            distances, indices = self.source_kdtree.query(target_points_3d)

            self.source_indices = indices
            self.distances = distances

        # Determine a reasonable threshold for identifying out-of-domain points
        if radius_of_influence is not None:
            self.distance_threshold = float(radius_of_influence)
        elif radius_of_influence is None:
            self.distance_threshold = float("inf")
        else:
            pass

    def _build_linear_interpolation(
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, radius_of_influence: float | None = None
    ) -> None:
        """Build Delaunay triangulation and interpolation structures for linear interpolation."""
        # Delaunay requires at least N+1 points in N dimensions. For 3D, we need at least 4 points.
        if len(source_points_3d) < 4:
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
            if len(source_points_3d) > 4:
                warnings.warn(
                    "Could not build Delaunay triangulation for linear interpolation:"
                    f" {e}. Falling back to nearest neighbor.",
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

    def _precompute_barycentric_weights(self, target_points_3d: np.ndarray, source_points_3d: np.ndarray) -> None:
        """Precompute barycentric weights for all target points."""
        # Initialize weights storage
        n_targets = len(target_points_3d)
        # Store weights and triangle indices for each target point
        self.precomputed_weights = {
            "simplex_indices": np.full(n_targets, -1, dtype=np.int32),
            "barycentric_weights": np.zeros((n_targets, 4), dtype=np.float64),
            "valid_points": np.zeros(n_targets, dtype=bool),
            "fallback_indices": np.full(n_targets, -1, dtype=np.int32),
            "type": "linear",
        }

        # We also maintain _fallback_indices as an attribute for backward compatibility
        self._fallback_indices = self.precomputed_weights["fallback_indices"]

        if self.triangles is None:
            msg = "Triangulation not initialized"
            raise RuntimeError(msg)

        # If fill_method is 'nearest', precompute all fallback indices first.
        if self.fill_method == "nearest" and self.source_kdtree is not None:
            _, fallback_indices = self.source_kdtree.query(target_points_3d)
            self.precomputed_weights["fallback_indices"] = fallback_indices
            self._fallback_indices = fallback_indices

        # Vectorized find_simplex call
        simplex_indices = self.triangles.find_simplex(target_points_3d, tol=-1e-8)

        # Retry with scaled points for those not found
        # This handles cases where points are slightly outside the convex hull (common on spheres)
        not_found_mask = simplex_indices == -1
        points_to_use = target_points_3d.copy()

        if np.any(not_found_mask):
            # Calculate centroid of source points
            centroid = np.mean(source_points_3d, axis=0)

            # Try a few scaling factors
            for scale in [0.999, 0.99, 0.95, 0.9]:
                # Only process points that haven't been found yet
                current_not_found_indices = np.where(simplex_indices == -1)[0]
                if len(current_not_found_indices) == 0:
                    break

                points_to_retry = target_points_3d[current_not_found_indices]

                # Scale points towards centroid
                vectors = points_to_retry - centroid
                scaled_points = centroid + vectors * scale

                # Try to find simplices for these scaled points
                retry_indices = self.triangles.find_simplex(scaled_points, tol=-1e-8)

                # Update indices and points where we found a simplex
                found_in_retry = retry_indices != -1
                if np.any(found_in_retry):
                    # Map back to original indices
                    original_indices = current_not_found_indices[found_in_retry]
                    simplex_indices[original_indices] = retry_indices[found_in_retry]
                    # Use scaled points for weight computation to ensure validity
                    points_to_use[original_indices] = scaled_points[found_in_retry]

        # For each target point, compute barycentric weights
        # Note: This loop is still Python but it runs only once during setup
        for target_idx, target_point in enumerate(points_to_use):
            simplex_idx = simplex_indices[target_idx]

            if simplex_idx != -1:
                # Point is inside a tetrahedron
                simplex_vertices = source_points_3d[self.triangles.simplices[simplex_idx]]
                weights = _compute_barycentric_weights_3d(target_point, simplex_vertices)
                if weights is not None:
                    self.precomputed_weights["simplex_indices"][target_idx] = simplex_idx
                    self.precomputed_weights["barycentric_weights"][target_idx] = weights
                    self.precomputed_weights["valid_points"][target_idx] = True
            else:
                # Point is outside the convex hull (even after scaling attempts)
                original_point = target_points_3d[target_idx]
                if self.fill_method == "nearest" and self.source_kdtree is not None:
                    self.precomputed_weights["simplex_indices"][target_idx] = -2  # Mark as nearest neighbor fallback
                    self.precomputed_weights["valid_points"][target_idx] = True
                    # Fallback index is already precomputed
                elif self.distance_threshold is not None and self.source_kdtree is not None:
                    distance, nearest_idx = self.source_kdtree.query(original_point)
                    if distance < self.distance_threshold:
                        self.precomputed_weights["simplex_indices"][target_idx] = -2
                        self.precomputed_weights["valid_points"][target_idx] = True
                        self.precomputed_weights["fallback_indices"][target_idx] = nearest_idx

    def interpolate(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Apply interpolation to source data.

        Args:
            source_data: Input data array with source grid dimensions
            use_precomputed: Whether to use precomputed weights (default True)

        Returns:
            Interpolated data on target grid
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
        """Perform conservative regridding."""
        if not use_precomputed or self.precomputed_weights is None:
            msg = "Weights not precomputed for conservative regridding."
            raise RuntimeError(msg)

        # source_data shape: (..., source_spatial_count)
        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        n_other_dims = len(original_shape) - 1

        if n_other_dims > 0:
            other_dims_size = int(np.prod(original_shape[:-1]))
            reshaped_data = source_data.reshape(other_dims_size, n_spatial)
        else:
            reshaped_data = source_data.reshape(1, n_spatial)

        # Get weights and indices
        source_indices = self.precomputed_weights["source_indices"]
        target_indices = self.precomputed_weights["target_indices"]
        weights = self.precomputed_weights["weights"]

        # Determine number of targets - this info is not explicitly in the sparse arrays
        # We need to know it from context or store it.
        # We can infer it from the target_indices max, but that might be smaller
        # than actual targets if last ones are empty. We should store n_targets
        # in precomputed_weights or pass it. For now, let's look at
        # self.target_points_3d if available.
        if self.target_points_3d is not None:
            n_targets = len(self.target_points_3d)
        elif "n_targets" in self.precomputed_weights:
            n_targets = self.precomputed_weights["n_targets"]
        else:
            # Fallback: max index + 1
            n_targets = int(target_indices.max()) + 1 if len(target_indices) > 0 else 0

        n_samples = reshaped_data.shape[0]

        # Apply weights using sparse matrix multiplication logic
        if HAS_NUMBA:
            # We need to update apply_weights_conservative to handle 1D arrays
            # Or assume we updated it.
            # Let's check if we updated it. We haven't yet.
            # We need to update _numba_kernels.py as well.
            result = apply_weights_conservative(reshaped_data, source_indices, target_indices, weights, n_targets)
        else:
            # Slow python fallback
            result = np.zeros((n_samples, n_targets), dtype=source_data.dtype)

            # This loop is slow for large arrays
            for k in range(len(weights)):
                t_idx = target_indices[k]
                s_idx = source_indices[k]
                w = weights[k]

                # Add contribution
                result[:, t_idx] += reshaped_data[:, s_idx] * w

        # Reshape back
        if n_other_dims > 0:
            target_shape = (*original_shape[:-1], n_targets)
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_structured(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform structured interpolation (bilinear/cubic)."""
        if not use_precomputed or self.precomputed_weights is None:
            msg = f"Weights not precomputed for {self.method} regridding."
            raise RuntimeError(msg)

        # source_data shape: (..., source_spatial_count)
        original_shape = source_data.shape
        n_spatial = original_shape[-1]
        n_other_dims = len(original_shape) - 1

        if n_other_dims > 0:
            other_dims_size = int(np.prod(original_shape[:-1]))
            reshaped_data = source_data.reshape(other_dims_size, n_spatial)
        else:
            reshaped_data = source_data.reshape(1, n_spatial)

        indices = self.precomputed_weights["indices"]
        weights = self.precomputed_weights["weights"]
        valid_mask = self.precomputed_weights["valid_mask"]

        if HAS_NUMBA:
            result = apply_weights_structured(reshaped_data, indices, weights, valid_mask)
        else:
            msg = "Numba required for structured interpolation"
            raise ImportError(msg)

        # Reshape back
        n_targets = indices.shape[0]
        if n_other_dims > 0:
            target_shape = (*original_shape[:-1], n_targets)
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_nearest(self, source_data: np.ndarray) -> np.ndarray:
        """Perform nearest neighbor interpolation."""
        # source_data shape: (..., source_spatial_count)
        # Reshape to handle multiple dimensions
        original_shape = source_data.shape
        n_spatial = original_shape[-1]  # Last dimension is spatial
        n_other_dims = len(original_shape) - 1

        if n_other_dims > 0:
            # Reshape to (other_dims_product, n_spatial)
            other_dims_size = int(np.prod(original_shape[:-1]))
            reshaped_data = source_data.reshape(other_dims_size, n_spatial)
        else:
            # Only spatial dimension
            reshaped_data = source_data.reshape(1, n_spatial)

        # Check for Numba acceleration availability
        if HAS_NUMBA:
            # Prepare arguments for Numba kernel
            if self.source_indices is None:
                msg = "Source indices not computed"
                raise RuntimeError(msg)

            # Determine valid mask
            if self.fill_method == "nan" and self.distances is not None and self.distance_threshold is not None:
                valid_mask = self.distances < self.distance_threshold
            else:
                valid_mask = np.ones(len(self.source_indices), dtype=bool)

            # Call Numba kernel
            # Note: We don't implement the complex "neighbor search fallback" in the Numba kernel yet
            # as it requires KDTree which isn't Numba-compatible.
            # But for the vast majority of points, this will be much faster.
            result = apply_weights_nearest(reshaped_data, self.source_indices, valid_mask)
        else:
            # Fallback to original implementation
            if self.source_indices is None:
                msg = "Source indices not computed"
                raise RuntimeError(msg)

            result = np.full((reshaped_data.shape[0], len(self.source_indices)), np.nan, dtype=source_data.dtype)

            # For each non-spatial slice
            for i in range(reshaped_data.shape[0]):
                slice_values = reshaped_data[i, :]

                if self.fill_method == "nan" and self.distances is not None and self.distance_threshold is not None:
                    # Only fill points that are within the domain (distance below threshold)
                    valid_mask = self.distances < self.distance_threshold
                    for j in range(len(valid_mask)):
                        if valid_mask[j]:
                            nearest_idx = self.source_indices[j]
                            if not np.isnan(slice_values[nearest_idx]):
                                result[i, j] = slice_values[nearest_idx]
                            # Simple fallback removed for brevity in non-Numba path comparison
                            # but original logic had complex fallback
                else:
                    # Fill all points with nearest neighbor values
                    for j in range(len(self.source_indices)):
                        nearest_idx = self.source_indices[j]
                        if not np.isnan(slice_values[nearest_idx]):
                            result[i, j] = slice_values[nearest_idx]

        # Reshape back to target shape
        if n_other_dims > 0:
            target_shape = (*original_shape[:-1], len(self.source_indices))
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_linear(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform linear interpolation using Delaunay triangulation."""
        if not use_precomputed or self.precomputed_weights is None:
            # Fallback to direct computation if precomputed weights not available
            warnings.warn("Precomputed weights not available, using direct computation", stacklevel=2)
            return self._interpolate_linear_direct(source_data)

        # source_data shape: (..., source_spatial_count)
        original_shape = source_data.shape
        n_spatial = original_shape[-1]  # Last dimension is spatial
        n_other_dims = len(original_shape) - 1

        if n_other_dims > 0:
            # Reshape to (other_dims_product, n_spatial)
            other_dims_size = int(np.prod(original_shape[:-1]))
            reshaped_data = source_data.reshape(other_dims_size, n_spatial)
        else:
            # Only spatial dimension
            reshaped_data = source_data.reshape(1, n_spatial)

        # Check for Numba acceleration availability
        if HAS_NUMBA:
            if self.triangles is None:
                # Should not happen if build_structures succeeded
                msg = "Triangulation not initialized"
                raise RuntimeError(msg)

            # Ensure we have the vertices cached as simple array
            if self._simplex_vertices_cache is None:
                self._simplex_vertices_cache = self.triangles.simplices.astype(np.int32)

            # Call Numba kernel
            result = apply_weights_linear(
                reshaped_data,
                self.precomputed_weights["simplex_indices"],
                self.precomputed_weights["barycentric_weights"],
                self.precomputed_weights["valid_points"],
                self._simplex_vertices_cache,
                self.precomputed_weights["fallback_indices"],
            )
        else:
            # Fallback to original implementation
            # Create result array
            n_targets = len(self.precomputed_weights["valid_points"])
            result = np.full((reshaped_data.shape[0], n_targets), np.nan, dtype=source_data.dtype)

            # Apply precomputed weights
            for target_idx in range(n_targets):
                if self.precomputed_weights["valid_points"][target_idx]:
                    simplex_idx = self.precomputed_weights["simplex_indices"][target_idx]

                    if simplex_idx >= 0:  # Valid tetrahedron found
                        # Get barycentric weights
                        weights = self.precomputed_weights["barycentric_weights"][target_idx]

                        # Get the source point indices for this triangle
                        if (
                            hasattr(self, "triangles")
                            and self.triangles is not None
                            and hasattr(self.triangles, "simplices")
                        ):
                            vertex_indices = self.triangles.simplices[simplex_idx]

                            for slice_idx in range(reshaped_data.shape[0]):
                                slice_values = reshaped_data[slice_idx, :]
                                vertex_values = slice_values[vertex_indices]
                                if np.any(np.isnan(vertex_values)):
                                    # Logic simplified for equivalence with Numba version
                                    if self.precomputed_weights["fallback_indices"][target_idx] != -1:
                                        fallback_idx = self.precomputed_weights["fallback_indices"][target_idx]
                                        result[slice_idx, target_idx] = slice_values[fallback_idx]
                                else:
                                    # Interpolate using barycentric weights
                                    result[slice_idx, target_idx] = np.dot(weights, vertex_values)

                    elif simplex_idx == -2:  # Fallback to nearest neighbor for points outside hull
                        if self.precomputed_weights["fallback_indices"][target_idx] != -1:
                            nearest_idx = self.precomputed_weights["fallback_indices"][target_idx]
                            for slice_idx in range(reshaped_data.shape[0]):
                                result[slice_idx, target_idx] = reshaped_data[slice_idx, nearest_idx]

        # Reshape back to target shape
        if n_other_dims > 0:
            target_shape = (*original_shape[:-1], len(self.precomputed_weights["valid_points"]))
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_linear_direct(self, _: np.ndarray) -> np.ndarray:
        """Direct computation of linear interpolation (fallback)."""
        # This is a fallback implementation if precomputed weights are not available
        # In practice, we should always have precomputed weights
        msg = (
            "Direct linear interpolation computation is not implemented. "
            "Use precomputed weights by calling build_structures first."
        )
        raise NotImplementedError(msg)
