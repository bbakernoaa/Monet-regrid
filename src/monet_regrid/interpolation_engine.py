"""
Optimized interpolation engine for curvilinear regridding.

This module implements efficient nearest neighbor and linear interpolation
with precomputed weights and sparse representations for memory optimization.

This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 [Your Organization].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications: Package renamed from xarray-regrid to monet-regrid,
URLs updated, and documentation adapted for new branding.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy.spatial import Delaunay, cKDTree  # type: ignore

try:
    from monet_regrid.methods._numba_kernels import apply_weights_linear, apply_weights_nearest
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available. Falling back to slower pure Python/NumPy implementation.")

try:
    from monet_regrid.methods._polygon_clipping import compute_conservative_weights
    HAS_POLYGON_CLIPPING = True
except ImportError:
    HAS_POLYGON_CLIPPING = False


class InterpolationEngine:
    """Optimized interpolation engine with precomputed weights."""

    def __init__(
        self,
        method: Literal["nearest", "linear", "conservative"] = "linear",
        spherical: bool = True,
        fill_method: Literal["nan", "nearest"] = "nan",
        extrapolate: bool = False,
    ):
        """Initialize the interpolation engine.

        Args:
            method: Interpolation method ('nearest', 'linear', or 'conservative')
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
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, radius_of_influence: float | None = None
    ) -> None:
        """Build interpolation structures based on method.

        Args:
            source_points_3d: Array of 3D source points (n, 3)
            target_points_3d: Array of 3D target points (m, 3)
            radius_of_influence: Maximum distance for valid interpolation
        """
        self.target_points_3d = target_points_3d
        if self.method == "nearest":
            self._build_nearest_neighbour(source_points_3d, target_points_3d, radius_of_influence)
        elif self.method == "linear":
            self._build_linear_interpolation(source_points_3d, target_points_3d, radius_of_influence)
        elif self.method == "conservative":
            # Conservative regridding requires boundaries, so this method shouldn't be called directly
            # with points. It should be called via build_conservative_structures.
            # However, if called, we can raise an error or fallback.
            raise ValueError(
                "Conservative regridding requires grid boundaries. "
                "Use build_conservative_structures() instead."
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}. Use 'nearest', 'linear', or 'conservative'")

    def build_conservative_structures(
        self,
        source_centers_3d: np.ndarray,
        target_centers_3d: np.ndarray,
        source_vertices_lonlat: np.ndarray,
        target_vertices_lonlat: np.ndarray,
        radius_of_influence: float | None = None
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
            raise ImportError("Numba is required for conservative regridding.")

        # 1. Build KDTree on source centers to find candidates
        self.source_kdtree = cKDTree(source_centers_3d)

        # 2. Query KDTree to find potential source cells for each target cell
        # We need a radius that ensures we cover the cell + some buffer.
        # Estimate cell size from vertices?
        # For now, use a heuristic or the provided radius.

        if radius_of_influence is None:
            # Heuristic: distance to furthest vertex from center?
            # Or just a safe large value if not provided.
            # Better: Calculate max cell radius from input
            # This is expensive. Let's assume a default or require it.
            # Let's assume typical global grid resolution ~100km -> 100,000m
            # Using 2x or 3x that covers most cases.
            radius_of_influence = 500000.0 # 500km

        # Find candidates
        # k=16 is usually enough for structured grids (overlapping 3x3 or 4x4 area)
        k_candidates = 25
        dists, indices = self.source_kdtree.query(target_centers_3d, k=k_candidates, distance_upper_bound=radius_of_influence)

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

        res_indices, res_weights = compute_conservative_weights(
            source_vertices_lonlat,
            target_vertices_lonlat,
            indices,
            counts
        )

        # 4. Store weights in sparse-friendly format
        # Filter out invalid weights (-1 indices or 0 weight)
        valid_mask = (res_indices != -1) & (res_weights > 0)

        # Check if we have any valid weights
        if not np.any(valid_mask):
            warnings.warn("Conservative regridding found no overlaps. Check coordinates or radius.")

        self.precomputed_weights = {
            "source_indices": res_indices,
            "weights": res_weights,
            "valid_mask": valid_mask,
            "type": "conservative"
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
        # If no radius is provided, calculate a default based on grid spacing
        elif source_points_3d.shape[0] > 1 and self.source_kdtree is not None:
            unique_points = np.unique(source_points_3d, axis=0)
            if unique_points.shape[0] > 1:
                # Find distance to 2nd nearest neighbor for each source point
                distances_to_2nd_neighbor, _ = self.source_kdtree.query(source_points_3d, k=2)
                # Use twice the mean distance to the 2nd nearest neighbor as a threshold
                if distances_to_2nd_neighbor.ndim == 2:
                    self.distance_threshold = float(2 * np.mean(distances_to_2nd_neighbor[:, 1]))
                else:
                    self.distance_threshold = float(2 * np.mean(distances_to_2nd_neighbor))
            else:
                self.distance_threshold = float("inf")
        else:
            self.distance_threshold = float("inf")

    def _build_linear_interpolation(
        self, source_points_3d: np.ndarray, target_points_3d: np.ndarray, radius_of_influence: float | None = None
    ) -> None:
        """Build Delaunay triangulation and interpolation structures for linear interpolation."""
        # Delaunay requires at least N+1 points in N dimensions. For 3D, we need at least 4 points.
        if len(source_points_3d) < 4:
            warnings.warn(
                "Linear interpolation requires at least 4 source points for robust 3D Delaunay triangulation. Falling back to nearest neighbor."
            )
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
                f"Could not build Delaunay triangulation for linear interpolation: {e}. Falling back to nearest neighbor."
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
            "type": "linear"
        }

        # We also maintain _fallback_indices as an attribute for backward compatibility
        self._fallback_indices = self.precomputed_weights["fallback_indices"]

        if self.triangles is None:
            raise RuntimeError("Triangulation not initialized")

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
                weights = self._compute_barycentric_weights_3d(target_point, simplex_vertices)
                if weights is not None:
                    self.precomputed_weights["simplex_indices"][target_idx] = simplex_idx
                    self.precomputed_weights["barycentric_weights"][target_idx] = weights
                    self.precomputed_weights["valid_points"][target_idx] = True
            else:
                # Point is outside the convex hull (even after scaling attempts)
                original_point = target_points_3d[target_idx]
                if self.fill_method == "nearest" and self.source_kdtree is not None:
                    _, nearest_idx = self.source_kdtree.query(original_point)
                    self.precomputed_weights["simplex_indices"][target_idx] = -2  # Mark as nearest neighbor fallback
                    self.precomputed_weights["valid_points"][target_idx] = True
                    self.precomputed_weights["fallback_indices"][target_idx] = nearest_idx
                elif self.distance_threshold is not None and self.source_kdtree is not None:
                    distance, nearest_idx = self.source_kdtree.query(original_point)
                    if distance < self.distance_threshold:
                        self.precomputed_weights["simplex_indices"][target_idx] = -2
                        self.precomputed_weights["valid_points"][target_idx] = True
                        self.precomputed_weights["fallback_indices"][target_idx] = nearest_idx

    def _point_in_tetrahedron(self, point: np.ndarray, tetra_vertices: np.ndarray) -> bool:
        """Check if a 3D point is contained in a tetrahedron."""
        weights = self._compute_barycentric_weights_3d(point, tetra_vertices)
        return bool(weights is not None and np.all(weights >= -1e-9) and np.all(weights <= 1 + 1e-9))

    def _compute_barycentric_weights_3d(self, point: np.ndarray, tetra_vertices: np.ndarray) -> np.ndarray | None:
        """Compute barycentric weights for a point in a 3D tetrahedron."""
        # Using the matrix inversion method
        T = np.vstack((tetra_vertices.T, np.ones(4)))
        p = np.append(point, 1)
        try:
            weights = np.linalg.solve(T, p)
            return weights
        except np.linalg.LinAlgError:
            # Singular matrix, likely degenerate tetrahedron
            return None

    def _point_in_triangle_3d(self, point: np.ndarray, triangle_vertices: np.ndarray) -> bool:
        """Check if a 3D point is contained in a triangle using barycentric coordinates."""
        # Legacy method
        v0 = triangle_vertices[1] - triangle_vertices[0]
        v1 = triangle_vertices[2] - triangle_vertices[0]
        v2 = point - triangle_vertices[0]

        # Calculate dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Calculate barycentric coordinates
        try:
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        except ZeroDivisionError:
            return False  # Degenerate triangle

        return bool((u >= 0) and (v >= 0) and (u + v <= 1))

    def _compute_barycentric_weights_2d_in_3d(self, point: np.ndarray, triangle_vertices: np.ndarray) -> np.ndarray:
        """Compute barycentric weights for a point in a 3D triangle."""
        # Legacy method
        v0 = triangle_vertices[1] - triangle_vertices[0]
        v1 = triangle_vertices[2] - triangle_vertices[0]
        v2 = point - triangle_vertices[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        try:
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            w = 1 - u - v  # Weight for first vertex
        except ZeroDivisionError:
            return np.array([1 / 3.0, 1 / 3.0, 1 / 3.0])

        return np.array([w, u, v])

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
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _interpolate_conservative(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform conservative regridding."""
        if not use_precomputed or self.precomputed_weights is None:
            raise RuntimeError("Weights not precomputed for conservative regridding.")

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
        weights = self.precomputed_weights["weights"]
        valid_mask = self.precomputed_weights["valid_mask"]

        n_targets = source_indices.shape[0]
        n_samples = reshaped_data.shape[0]

        result = np.zeros((n_samples, n_targets), dtype=source_data.dtype)

        # Apply weights (Manual sparse matrix multiplication)
        # Numba-optimize this too for speed?
        # A simple python loop over targets is fine if max_overlaps is small
        # But for n_targets=1M, python loop is slow.
        # Let's add a kernel for this.

        if HAS_NUMBA:
            from monet_regrid.methods._numba_kernels import apply_weights_conservative
            result = apply_weights_conservative(reshaped_data, source_indices, weights, valid_mask)
        else:
            # Slow python fallback
            for t_idx in range(n_targets):
                indices_t = source_indices[t_idx]
                weights_t = weights[t_idx]
                mask_t = valid_mask[t_idx]

                # Sum (val * weight) for all overlaps
                for k in range(len(indices_t)):
                    if mask_t[k]:
                        idx = indices_t[k]
                        w = weights_t[k]
                        result[:, t_idx] += reshaped_data[:, idx] * w

        # Reshape back
        if n_other_dims > 0:
            target_shape = original_shape[:-1] + (n_targets,)
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
                raise RuntimeError("Source indices not computed")

            # Determine valid mask
            if self.fill_method == "nan" and self.distances is not None and self.distance_threshold is not None:
                valid_mask = self.distances < self.distance_threshold
            else:
                valid_mask = np.ones(len(self.source_indices), dtype=bool)

            # Call Numba kernel
            # Note: We don't implement the complex "neighbor search fallback" in the Numba kernel yet
            # as it requires KDTree which isn't Numba-compatible.
            # But for the vast majority of points, this will be much faster.
            result = apply_weights_nearest(
                reshaped_data,
                self.source_indices,
                valid_mask
            )
        else:
            # Fallback to original implementation
            if self.source_indices is None:
                raise RuntimeError("Source indices not computed")

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
            target_shape = original_shape[:-1] + (len(self.source_indices),)
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_linear(self, source_data: np.ndarray, use_precomputed: bool = True) -> np.ndarray:
        """Perform linear interpolation using Delaunay triangulation."""
        if not use_precomputed or self.precomputed_weights is None:
            # Fallback to direct computation if precomputed weights not available
            warnings.warn("Precomputed weights not available, using direct computation")
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
                raise RuntimeError("Triangulation not initialized")

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
                self.precomputed_weights["fallback_indices"]
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
            target_shape = original_shape[:-1] + (len(self.precomputed_weights["valid_points"]),)
            return result.reshape(target_shape)
        else:
            return result.reshape(-1)

    def _interpolate_linear_direct(self, source_data: np.ndarray) -> np.ndarray:
        """Direct computation of linear interpolation (fallback)."""
        # This is a fallback implementation if precomputed weights are not available
        # In practice, we should always have precomputed weights
        raise NotImplementedError(
            "Direct linear interpolation computation is not implemented. "
            "Use precomputed weights by calling build_structures first."
        )
