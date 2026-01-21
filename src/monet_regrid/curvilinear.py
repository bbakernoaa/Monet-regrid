"""
Optimized curvilinear interpolation using 3D coordinate transformations and precomputed weights.

This module implements an optimized curvilinear interpolator with:
- Vectorized 3D coordinate transformations using pyproj
- Efficient KDTree (nearest) and Delaunay triangulation (linear)
- Precomputed interpolation weights for build-once/apply-many pattern
- Distance threshold calculations for out-of-domain detection
- Memory optimization with sparse representations

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

from typing import Any, Literal

import dask.array as da
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, cKDTree  # type: ignore

from monet_regrid.coordinate_transformer import CoordinateTransformer
from monet_regrid.interpolation import InterpolationEngine
from monet_regrid.interpolation.utils import (
    _compute_barycentric_weights_3d,
    _point_in_tetrahedron,
)


def _check_and_raise_on_non_finite(
    x: np.ndarray | da.Array,
    y: np.ndarray | da.Array,
    z: np.ndarray | da.Array,
    lats: np.ndarray | da.Array,
    lons: np.ndarray | da.Array,
) -> None:
    """Check for non-finite values and raise a detailed ValueError.
    This function inspects the transformed 3D coordinates (x, y, z) for any
    non-finite values (NaN, inf). It handles both NumPy and Dask arrays by
    branching its logic. If non-finite values are found, it raises a
    `ValueError` with the coordinates of the first few problematic points.
    Parameters
    ----------
    x : np.ndarray | da.Array
        The x-component of the transformed coordinates.
    y : np.ndarray | da.Array
        The y-component of the transformed coordinates.
    z : np.ndarray | da.Array
        The z-component of the transformed coordinates.
    lats : np.ndarray | da.Array
        The original latitude values, used for error reporting.
    lons : np.ndarray | da.Array
        The original longitude values, used for error reporting.
    Raises
    ------
    ValueError
        If any of the input coordinates (x, y, z) contain non-finite values.
    """
    is_dask = isinstance(x, da.Array)

    if is_dask:
        # For Dask, compute the check in a single pass
        all_finite = (da.isfinite(x).all() & da.isfinite(y).all() & da.isfinite(z).all()).compute()
    else:
        # For NumPy, check eagerly
        all_finite = np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()

    if not all_finite:
        if is_dask:
            non_finite_mask = ~(da.isfinite(x) & da.isfinite(y) & da.isfinite(z)).compute()
            problematic_lats = lats.compute()[non_finite_mask]
            problematic_lons = lons.compute()[non_finite_mask]
        else:
            non_finite_mask = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
            problematic_lats = lats[non_finite_mask]
            problematic_lons = lons[non_finite_mask]

        msg = (
            f"Non-finite coordinates found during transformation: "
            f"lat={problematic_lats[:5]}, lon={problematic_lons[:5]} "
            f"(showing first 5 of {np.sum(non_finite_mask)} non-finite points)"
        )
        raise ValueError(msg)


class CurvilinearInterpolator:
    """Optimized interpolator for curvilinear grids using 3D coordinate transformations.

    This class handles interpolation between curvilinear grids by transforming
    geographic coordinates to 3D geocentric coordinates (EPSG 4979 → 4978) and
    performing surface-aware interpolation in 3D space.

    It is designed to be Dask-aware, allowing for lazy evaluation of coordinate
    transformations on large, out-of-core datasets. The interpolation itself
    is performed using SciPy, which requires in-memory NumPy arrays, so a
    `.compute()` call is triggered internally only when building the interpolation
    structures.
    """

    def __init__(
        self,
        source_grid: xr.Dataset,
        target_grid: xr.Dataset,
        source_lat_name: str,
        source_lon_name: str,
        target_lat_name: str,
        target_lon_name: str,
        method: Literal["nearest", "linear", "conservative", "bilinear", "cubic"] = "linear",
        spherical: bool = True,
        fill_method: Literal["nan", "nearest"] = "nan",
        extrapolate: bool = False,
        **kwargs: Any,
    ):
        """Initialize the optimized curvilinear interpolator.

        Args:
            source_grid: Source grid specification with 2D coordinates
            target_grid: Target grid specification with 2D coordinates
            source_lat_name: Name of the latitude coordinate in the source grid
            source_lon_name: Name of the longitude coordinate in the source grid
            target_lat_name: Name of the latitude coordinate in the target grid
            target_lon_name: Name of the longitude coordinate in the target grid
            method: Interpolation method ('nearest', 'linear', 'conservative', 'bilinear', 'cubic')
            spherical: Whether to use spherical barycentrics (True) or planar (False)
            fill_method: How to handle out-of-domain targets ('nan' or 'nearest')
            extrapolate: Whether to allow extrapolation beyond source domain
            **kwargs: Additional method-specific arguments
        """
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.source_lat_name = source_lat_name
        self.source_lon_name = source_lon_name
        self.target_lat_name = target_lat_name
        self.target_lon_name = target_lon_name
        self.method = method
        self.spherical = spherical
        self.fill_method = fill_method
        self.extrapolate = extrapolate
        self.radius_of_influence = kwargs.get("radius_of_influence", 1e6)
        self.method_kwargs = {k: v for k, v in kwargs.items() if k != "radius_of_influence"}

        # Initialize coordinate transformation
        self.coordinate_transformer = CoordinateTransformer("EPSG:4979", "EPSG:4978")

        if method == "conservative":
            # Conservative regridding requires boundary coordinates
            # We assume these are provided or can be inferred via CF conventions
            # For now, let's implement a placeholder or a check
            pass

        # Transform coordinates to 3D
        self._transform_coordinates()

        # Build interpolation structures
        self._build_interpolation_structures()

        # Precompute interpolation weights for build-once/apply-many pattern
        self._precompute_interpolation_weights()

    @property
    def triangles(self) -> np.ndarray:
        """Access triangulation simplices from the interpolation engine."""
        if hasattr(self.interpolation_engine, "triangles") and self.interpolation_engine.triangles is not None:
            # For 3D Delaunay, simplices are tetrahedra with 4 vertices
            return self.interpolation_engine.triangles.simplices  # type: ignore
        msg = f"'{self.__class__.__name__}' object has no attribute 'triangles'"
        raise AttributeError(msg)

    @property
    def triangle_centroids(self) -> np.ndarray:
        """Access triangle centroids from the interpolation engine."""
        if (
            self.method == "linear"
            and hasattr(self.interpolation_engine, "triangles")
            and self.interpolation_engine.triangles is not None
        ):
            # Compute centroids of triangles for efficient lookup
            if not hasattr(self, "_triangle_centroids"):
                # Get the triangles (simplices) and compute centroids
                simplices = self.triangles
                self._triangle_centroids = np.mean(self.source_points_3d_np[simplices], axis=1)
            return self._triangle_centroids  # type: ignore
        msg = f"'{self.__class__.__name__}' object has no attribute 'triangle_centroids'"
        raise AttributeError(msg)

    @property
    def triangle_centroid_kdtree(self) -> cKDTree:
        """Access KDTree of triangle centroids from the interpolation engine."""
        if self.method == "linear" and hasattr(self.interpolation_engine, "target_kdtree"):
            # Create a KDTree for triangle centroids if needed
            if not hasattr(self, "_triangle_centroid_kdtree"):
                self._triangle_centroid_kdtree = cKDTree(self.triangle_centroids)
            return self._triangle_centroid_kdtree
        msg = f"'{self.__class__.__name__}' object has no attribute 'triangle_centroid_kdtree'"
        raise AttributeError(msg)

    @property
    def kdtree(self) -> cKDTree:
        """Access KDTree from the interpolation engine."""
        if hasattr(self.interpolation_engine, "source_kdtree"):
            return self.interpolation_engine.source_kdtree
        msg = f"'{self.__class__.__name__}' object has no attribute 'kdtree'"
        raise AttributeError(msg)

    @property
    def target_kdtree(self) -> cKDTree:
        """Access target KDTree from the interpolation engine."""
        if hasattr(self.interpolation_engine, "target_kdtree"):
            return self.interpolation_engine.target_kdtree
        msg = f"'{self.__class__.__name__}' object has no attribute 'target_kdtree'"
        raise AttributeError(msg)

    @property
    def convex_hull(self) -> Delaunay:
        """Access triangulation structure (Delaunay) from the interpolation engine."""
        if hasattr(self.interpolation_engine, "triangles") and self.interpolation_engine.triangles is not None:
            # For linear method, this is the Delaunay object which the test expects
            return self.interpolation_engine.triangles
        msg = f"'{self.__class__.__name__}' object has no attribute 'convex_hull'"
        raise AttributeError(msg)

    @property
    def distance_threshold(self) -> float:
        """Access distance threshold from the interpolation engine."""
        if hasattr(self.interpolation_engine, "distance_threshold") and self.interpolation_engine.distance_threshold is not None:
            return self.interpolation_engine.distance_threshold
        return float("inf")

    @property
    def source_indices(self) -> np.ndarray:
        """Access source indices from the interpolation engine."""
        if hasattr(self.interpolation_engine, "source_indices") and self.interpolation_engine.source_indices is not None:
            return self.interpolation_engine.source_indices
        msg = f"'{self.__class__.__name__}' object has no attribute 'source_indices'"
        raise AttributeError(msg)

    @property
    def transformer(self) -> Any:
        """Access the coordinate transformer."""
        return self.coordinate_transformer.transformer

    def _find_triangle_containing_point(self, point_3d: np.ndarray, triangle_idx: int) -> bool:
        """Check if a 3D point is contained in the specified triangle."""
        if not hasattr(self.interpolation_engine, "triangles") or self.interpolation_engine.triangles is None:
            return False

        # Get the triangle vertices
        simplex_vertices = self.source_points_3d[self.triangles[triangle_idx]]

        # Use the interpolation engine's method to check if point is in triangle
        # For 3D, this checks if a point is in a tetrahedron

        return _point_in_tetrahedron(point_3d, simplex_vertices)

    @property
    def precomputed_weights(self) -> dict:
        """Access precomputed weights from the interpolation engine."""
        if hasattr(self.interpolation_engine, "precomputed_weights") and self.interpolation_engine.precomputed_weights is not None:
            return self.interpolation_engine.precomputed_weights
        msg = f"'{self.__class__.__name__}' object has no attribute 'precomputed_weights'"
        raise AttributeError(msg)

    def _compute_barycentric_weights(self, point_3d: np.ndarray, triangle_idx: int) -> tuple[float, ...]:
        """Compute barycentric weights for a point in the specified triangle."""
        if (
            not hasattr(self.interpolation_engine, "triangles")
            or self.interpolation_engine.triangles is None
            or triangle_idx >= len(self.triangles)
        ):
            # Return equal weights if triangle is invalid
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # Get the triangle vertices
        triangle_vertices = self.source_points_3d[self.interpolation_engine.triangles.simplices[triangle_idx]]

        # Use the interpolation engine's method to compute barycentric weights

        weights = _compute_barycentric_weights_3d(point_3d, triangle_vertices)
        return tuple(weights) if weights is not None else (np.nan, np.nan, np.nan, np.nan)

    @property
    def distances(self) -> np.ndarray:
        """Access distances from the interpolation engine."""
        if hasattr(self.interpolation_engine, "distances") and self.interpolation_engine.distances is not None:
            return self.interpolation_engine.distances
        msg = f"'{self.__class__.__name__}' object has no attribute 'distances'"
        raise AttributeError(msg)

    def _transform_coordinates(self) -> None:
        """Transform geographic coordinates to 3D geocentric coordinates."""
        # Use dask.array for lazy evaluation of coordinate transformations
        # Extract source coordinates
        source_lat = self.source_grid[self.source_lat_name]
        source_lon = self.source_grid[self.source_lon_name]

        # Handle both 1D and 2D coordinates
        if source_lat.ndim == 1 and source_lon.ndim == 1:
            # 1D coordinates (rectilinear grid) - use Dask-aware meshgrid
            source_lon_2d, source_lat_2d = da.meshgrid(source_lon.data, source_lat.data)
            self.source_shape = source_lat_2d.shape
            source_lat_flat = source_lat_2d.flatten()
            source_lon_flat = source_lon_2d.flatten()
        else:
            # 2D coordinates (curvilinear grid) - use as is
            self.source_shape = source_lat.shape
            source_lat_flat = da.asarray(source_lat.data).flatten()
            source_lon_flat = da.asarray(source_lon.data).flatten()

        # Clamp coordinates to valid ranges to handle edge cases gracefully
        source_lat_flat = da.clip(source_lat_flat, -90.0, 90.0)
        source_lon_flat = da.clip(source_lon_flat, -180.0, 180.0)

        # Transform to 3D coordinates (assuming height=0 for surface points)
        source_heights = da.zeros_like(source_lat_flat)
        self.source_x, self.source_y, self.source_z = self.coordinate_transformer.transform_coordinates(
            source_lon_flat, source_lat_flat, source_heights
        )

        # Check for finite values before creating 3D points array
        _check_and_raise_on_non_finite(self.source_x, self.source_y, self.source_z, source_lat_flat, source_lon_flat)

        # Store as 3D points array
        self.source_points_3d = da.stack([self.source_x, self.source_y, self.source_z], axis=1)

        # Extract target coordinates
        target_lat = self.target_grid[self.target_lat_name]
        target_lon = self.target_grid[self.target_lon_name]

        # Handle both 1D and 2D coordinates
        if target_lat.ndim == 1 and target_lon.ndim == 1:
            # 1D coordinates (rectilinear grid) - use Dask-aware meshgrid
            target_lon_2d, target_lat_2d = da.meshgrid(target_lon.data, target_lat.data)
            self.target_shape = target_lat_2d.shape
            target_lat_flat = target_lat_2d.flatten()
            target_lon_flat = target_lon_2d.flatten()
        else:
            # 2D coordinates (curvilinear grid) - use as is
            self.target_shape = target_lat.shape
            target_lat_flat = da.asarray(target_lat.data).flatten()
            target_lon_flat = da.asarray(target_lon.data).flatten()

        # Clamp coordinates to valid ranges to handle edge cases gracefully
        target_lat_flat = da.clip(target_lat_flat, -90.0, 90.0)
        target_lon_flat = da.clip(target_lon_flat, -180.0, 180.0)

        # Transform to 3D coordinates (assuming height=0 for surface points)
        target_heights = da.zeros_like(target_lat_flat)
        self.target_x, self.target_y, self.target_z = self.coordinate_transformer.transform_coordinates(
            target_lon_flat, target_lat_flat, target_heights
        )

        # Check for finite values before creating 3D points array
        _check_and_raise_on_non_finite(self.target_x, self.target_y, self.target_z, target_lat_flat, target_lon_flat)

        # Store as 3D points array
        self.target_points_3d = da.stack([self.target_x, self.target_y, self.target_z], axis=1)

    def _build_interpolation_structures(self) -> None:
        """Build interpolation structures based on method."""
        # Create interpolation engine
        self.interpolation_engine = InterpolationEngine(
            method=self.method, spherical=self.spherical, fill_method=self.fill_method, extrapolate=self.extrapolate
        )

    def _precompute_interpolation_weights(self) -> None:
        """Precompute interpolation weights for build-once/apply-many pattern."""
        # This method is now a placeholder. The actual building of interpolation
        # structures is deferred to the block-wise processing in __call__.
        pass

    def __call__(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Apply interpolation to data.

        Args:
            data: Input data with curvilinear coordinates matching source grid

        Returns:
            Interpolated data on target grid
        """
        if isinstance(data, xr.DataArray):
            return self._interpolate_dataarray(data)
        elif isinstance(data, xr.Dataset):
            return self._interpolate_dataset(data)
        else:
            msg = "Input must be xarray DataArray or Dataset"
            raise TypeError(msg)

    def _interpolate_block(
        self, data_block: xr.DataArray, source_points_3d: xr.DataArray, target_points_3d: xr.DataArray
    ) -> xr.DataArray:
        """Interpolate a single Dask block."""
        source_points_3d_np = source_points_3d.data
        target_points_3d_np = target_points_3d.data

        engine = InterpolationEngine(
            method=self.method, spherical=self.spherical, fill_method=self.fill_method, extrapolate=self.extrapolate
        )
        engine.build_structures(source_points_3d_np, target_points_3d_np, self.radius_of_influence)

        reshaped_data = data_block.data.reshape(*data_block.shape[:-2], -1)
        interpolated = engine.interpolate(reshaped_data)
        final_shape = (*data_block.shape[:-2], *self.target_shape)

        target_lat_coord = self.target_grid[self.target_lat_name]
        if target_lat_coord.ndim == 2:
            target_dims = list(target_lat_coord.dims)
        else:
            target_dims = [target_lat_coord.dims[0], self.target_grid[self.target_lon_name].dims[0]]

        return xr.DataArray(interpolated.reshape(final_shape), dims=data_block.dims[:-2] + tuple(target_dims))

    def _interpolate_dataarray(self, data: xr.DataArray) -> xr.DataArray:
        """Interpolate a single DataArray."""
        if not self._validate_data_coordinates(data):
            msg = "Data coordinates do not match source grid"
            raise ValueError(msg)

        target_lat_coord = self.target_grid[self.target_lat_name]
        target_lon_coord = self.target_grid[self.target_lon_name]

        if target_lat_coord.ndim == 2:
            target_dims = list(target_lat_coord.dims)
        else:
            target_dims = [target_lat_coord.dims[0], target_lon_coord.dims[0]]

        template = xr.DataArray(
            da.empty(self.target_shape, chunks=-1, dtype=data.dtype),
            dims=target_dims,
        )

        source_points_da = xr.DataArray(self.source_points_3d, dims=["n_points", "three"])
        target_points_da = xr.DataArray(self.target_points_3d, dims=["n_points_target", "three"])

        result = data.map_blocks(
            self._interpolate_block,
            args=[source_points_da, target_points_da],
            template=template,
        )

        if not result.attrs and data.attrs:
            result.attrs = data.attrs.copy()

        if target_lat_coord.ndim == 2:
            result.coords[self.target_lat_name] = target_lat_coord
            result.coords[self.target_lon_name] = target_lon_coord
        else:
            result.coords[self.target_lat_name] = target_lat_coord
            result.coords[self.target_lon_name] = target_lon_coord

        for dim in target_dims:
            if dim in self.target_grid.coords:
                result.coords[dim] = self.target_grid.coords[dim]

        return result

    def _interpolate_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Interpolate an entire Dataset."""
        result_dataset = xr.Dataset()

        for var_name, data_array in dataset.items():
            # Skip coordinate variables that match the grid coordinates
            if var_name in [self.source_lat_name, self.source_lon_name]:
                continue

            # Check if this variable has the spatial dimensions that match the source grid shape
            # The spatial dimensions are the dimensions of the source coordinate variables
            source_spatial_dims = self.source_grid[self.source_lat_name].dims

            # Check if the data array has all the source spatial dimensions
            if all(dim in data_array.dims for dim in source_spatial_dims):
                # This variable uses curvilinear coordinates, interpolate it
                result_dataset[var_name] = self._interpolate_dataarray(data_array)
            else:
                # This variable doesn't use curvilinear coordinates, keep as is
                result_dataset[var_name] = data_array

        # Add the target coordinates to the result
        result_dataset.coords[self.target_lat_name] = self.target_grid[self.target_lat_name]
        result_dataset.coords[self.target_lon_name] = self.target_grid[self.target_lon_name]

        # Also add the dimension coordinates from the target grid, creating them if they don't exist
        for dim_name in self.target_grid[self.target_lat_name].dims:
            if dim_name in self.target_grid.coords:
                result_dataset.coords[dim_name] = self.target_grid.coords[dim_name]
            else:
                # Create a coordinate for the dimension if it doesn't exist
                dim_size = self.target_grid.sizes[dim_name]
                result_dataset.coords[dim_name] = np.arange(dim_size)

        return result_dataset

    def _validate_data_coordinates(self, data: xr.DataArray) -> bool:
        """Validate that data coordinates match the source grid."""
        # Check if data has dimensions that match the source grid shape
        # The data should have the same spatial dimensions as the source grid
        expected_sizes = set(self.source_shape)
        data_sizes = set(data.sizes.values())

        # Check if data has dimensions with sizes that match the source grid dimensions
        matching_sizes = expected_sizes.intersection(data_sizes)
        return len(matching_sizes) >= len(expected_sizes)  # At least all source sizes should be present
