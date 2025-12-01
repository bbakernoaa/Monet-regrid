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

import abc
from typing import Any, Literal, Tuple

import numpy as np
import pyproj
import xarray as xr
from scipy.spatial import Delaunay, cKDTree  # type: ignore

from .coordinate_transformer import CoordinateTransformer
from .interpolation_engine import InterpolationEngine


class CurvilinearInterpolator:
    """Optimized interpolator for curvilinear grids using 3D coordinate transformations.

    This class handles interpolation between curvilinear grids by transforming
    geographic coordinates to 3D geocentric coordinates (EPSG 4979 → 4978) and
    performing surface-aware interpolation in 3D space.
    """

    def __init__(
        self,
        source_grid: xr.Dataset,
        target_grid: xr.Dataset,
        method: Literal["nearest", "linear"] = "linear",
        spherical: bool = True,
        fill_method: Literal["nan", "nearest"] = "nan",
        extrapolate: bool = False,
        **kwargs: Any,
    ):
        """Initialize the optimized curvilinear interpolator.

        Args:
            source_grid: Source grid specification with 2D coordinates
            target_grid: Target grid specification with 2D coordinates
            method: Interpolation method ('nearest' or 'linear')
            spherical: Whether to use spherical barycentrics (True) or planar (False)
            fill_method: How to handle out-of-domain targets ('nan' or 'nearest')
            extrapolate: Whether to allow extrapolation beyond source domain
            **kwargs: Additional method-specific arguments
        """
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.method = method
        self.spherical = spherical
        self.fill_method = fill_method
        self.extrapolate = extrapolate
        self.radius_of_influence = kwargs.get("radius_of_influence", 1e6)
        self.method_kwargs = {k: v for k, v in kwargs.items() if k != "radius_of_influence"}

        # Initialize coordinate transformation
        self.coordinate_transformer = CoordinateTransformer("EPSG:4979", "EPSG:4978")

        # Extract and validate coordinates
        self._validate_coordinates()

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
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'triangles'")

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
                self._triangle_centroids = np.mean(self.source_points_3d[simplices], axis=1)
            return self._triangle_centroids  # type: ignore
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'triangle_centroids'")

    @property
    def triangle_centroid_kdtree(self) -> cKDTree:
        """Access KDTree of triangle centroids from the interpolation engine."""
        if self.method == "linear" and hasattr(self.interpolation_engine, "target_kdtree"):
            # Create a KDTree for triangle centroids if needed
            if not hasattr(self, "_triangle_centroid_kdtree"):
                self._triangle_centroid_kdtree = cKDTree(self.triangle_centroids)
            return self._triangle_centroid_kdtree
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'triangle_centroid_kdtree'")

    @property
    def kdtree(self) -> cKDTree:
        """Access KDTree from the interpolation engine."""
        if hasattr(self.interpolation_engine, "source_kdtree"):
            return self.interpolation_engine.source_kdtree
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'kdtree'")

    @property
    def target_kdtree(self) -> cKDTree:
        """Access target KDTree from the interpolation engine."""
        if hasattr(self.interpolation_engine, "target_kdtree"):
            return self.interpolation_engine.target_kdtree
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'target_kdtree'")

    @property
    def convex_hull(self) -> Delaunay:
        """Access triangulation structure (Delaunay) from the interpolation engine."""
        if hasattr(self.interpolation_engine, "triangles") and self.interpolation_engine.triangles is not None:
            # For linear method, this is the Delaunay object which the test expects
            return self.interpolation_engine.triangles
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'convex_hull'")

    @property
    def distance_threshold(self) -> float:
        """Access distance threshold from the interpolation engine."""
        if (
            hasattr(self.interpolation_engine, "distance_threshold")
            and self.interpolation_engine.distance_threshold is not None
        ):
            return self.interpolation_engine.distance_threshold
        return float("inf")

    @property
    def source_indices(self) -> np.ndarray:
        """Access source indices from the interpolation engine."""
        if (
            hasattr(self.interpolation_engine, "source_indices")
            and self.interpolation_engine.source_indices is not None
        ):
            return self.interpolation_engine.source_indices
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'source_indices'")

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
        return self.interpolation_engine._point_in_tetrahedron(point_3d, simplex_vertices)

    @property
    def precomputed_weights(self) -> dict:
        """Access precomputed weights from the interpolation engine."""
        if (
            hasattr(self.interpolation_engine, "precomputed_weights")
            and self.interpolation_engine.precomputed_weights is not None
        ):
            return self.interpolation_engine.precomputed_weights
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'precomputed_weights'")

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
        weights = self.interpolation_engine._compute_barycentric_weights_3d(point_3d, triangle_vertices)
        return tuple(weights) if weights is not None else (np.nan, np.nan, np.nan, np.nan)

    @property
    def distances(self) -> np.ndarray:
        """Access distances from the interpolation engine."""
        if hasattr(self.interpolation_engine, "distances") and self.interpolation_engine.distances is not None:
            return self.interpolation_engine.distances
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'distances'")

    def _validate_coordinates(self) -> None:
        """Validate that source and target grids have latitude and longitude coordinates."""
        # Use cf-xarray to find latitude and longitude coordinates in source grid
        try:
            source_lat = self.source_grid.cf["latitude"]
            source_lon = self.source_grid.cf["longitude"]
            self.source_lat_name = source_lat.name
            self.source_lon_name = source_lon.name
        except KeyError:
            # Fallback to manual search if cf-xarray fails
            lat_coords = [
                name
                for name in self.source_grid.coords
                if "lat" in str(name).lower() or "latitude" in str(name).lower()
            ]
            lon_coords = [
                name
                for name in self.source_grid.coords
                if "lon" in str(name).lower() or "longitude" in str(name).lower()
            ]

            if not lat_coords or not lon_coords:
                raise ValueError("Source grid must have latitude and longitude coordinates")

            # Use the first found coordinate name
            self.source_lat_name = lat_coords[0]
            self.source_lon_name = lon_coords[0]

        # Use cf-xarray to find latitude and longitude coordinates in target grid
        try:
            target_lat = self.target_grid.cf["latitude"]
            target_lon = self.target_grid.cf["longitude"]
            self.target_lat_name = target_lat.name
            self.target_lon_name = target_lon.name
        except KeyError:
            # Fallback to manual search if cf-xarray fails
            lat_coords = [
                name
                for name in self.target_grid.coords
                if "lat" in str(name).lower() or "latitude" in str(name).lower()
            ]
            lon_coords = [
                name
                for name in self.target_grid.coords
                if "lon" in str(name).lower() or "longitude" in str(name).lower()
            ]

            if not lat_coords or not lon_coords:
                raise ValueError("Target grid must have latitude and longitude coordinates")

            # Use the first found coordinate name
            self.target_lat_name = lat_coords[0]
            self.target_lon_name = lon_coords[0]

        # Validate source coordinates - allow both 1D (rectilinear) and 2D (curvilinear)
        source_lat_data = self.source_grid[self.source_lat_name]
        source_lon_data = self.source_grid[self.source_lon_name]

        # Allow both 1D (rectilinear) and 2D (curvilinear) coordinates for source
        if source_lat_data.ndim not in [1, 2] or source_lon_data.ndim not in [1, 2]:
            raise ValueError(
                f"Source coordinates must be 1D or 2D. Got lat={source_lat_data.ndim}D, lon={source_lon_data.ndim}D"
            )

        if source_lat_data.ndim != source_lon_data.ndim:
            raise ValueError(
                f"Source latitude and longitude coordinates must have same number of dimensions. "
                f"Got lat={source_lat_data.ndim}D, lon={source_lon_data.ndim}D"
            )

        target_lat_data = self.target_grid[self.target_lat_name]
        target_lon_data = self.target_grid[self.target_lon_name]

        # Allow both 1D (rectilinear) and 2D (curvilinear) for target, but they must match
        if target_lat_data.ndim not in [1, 2] or target_lon_data.ndim not in [1, 2]:
            raise ValueError(
                f"Target coordinates must be 1D or 2D. Got lat={target_lat_data.ndim}D, lon={target_lon_data.ndim}D"
            )

        if target_lat_data.ndim != target_lon_data.ndim:
            raise ValueError(
                f"Target latitude and longitude coordinates must have same number of dimensions. "
                f"Got lat={target_lat_data.ndim}D, lon={target_lon_data.ndim}D"
            )

    def _transform_coordinates(self) -> None:
        """Transform geographic coordinates to 3D geocentric coordinates."""
        # Extract source coordinates
        source_lat = self.source_grid[self.source_lat_name]
        source_lon = self.source_grid[self.source_lon_name]

        # Handle both 1D and 2D coordinates
        if source_lat.ndim == 1 and source_lon.ndim == 1:
            # 1D coordinates (rectilinear grid) - need to create 2D meshgrid
            source_lon_2d, source_lat_2d = np.meshgrid(source_lon.data, source_lat.data)
            self.source_shape = source_lat_2d.shape
            source_lat_flat = source_lat_2d.flatten()
            source_lon_flat = source_lon_2d.flatten()
        else:
            # 2D coordinates (curvilinear grid) - use as is
            self.source_shape = source_lat.shape
            source_lat_flat = source_lat.data.flatten()
            source_lon_flat = source_lon.data.flatten()

        # Clamp coordinates to valid ranges to handle edge cases gracefully
        source_lat_flat = np.clip(source_lat_flat, -90.0, 90.0)
        source_lon_flat = np.clip(source_lon_flat, -180.0, 180.0)

        # Transform to 3D coordinates (assuming height=0 for surface points)
        source_heights = np.zeros_like(source_lat_flat)
        self.source_x, self.source_y, self.source_z = self.coordinate_transformer.transform_coordinates(
            source_lon_flat, source_lat_flat, source_heights
        )

        # Check for finite values before creating 3D points array
        if not (
            np.isfinite(self.source_x).all() and np.isfinite(self.source_y).all() and np.isfinite(self.source_z).all()
        ):
            # Identify problematic coordinates
            non_finite_mask = ~(np.isfinite(self.source_x) & np.isfinite(self.source_y) & np.isfinite(self.source_z))
            if np.any(non_finite_mask):
                problematic_lats = source_lat_flat[non_finite_mask]
                problematic_lons = source_lon_flat[non_finite_mask]
                raise ValueError(
                    f"Non-finite coordinates found during transformation: "
                    f"lat={problematic_lats[:5]}, lon={problematic_lons[:5]} "
                    f"(showing first 5 of {np.sum(non_finite_mask)} non-finite points)"
                )

        # Store as 3D points array
        self.source_points_3d = np.column_stack([self.source_x, self.source_y, self.source_z])

        # Extract target coordinates
        target_lat = self.target_grid[self.target_lat_name]
        target_lon = self.target_grid[self.target_lon_name]

        # Handle both 1D and 2D coordinates
        if target_lat.ndim == 1 and target_lon.ndim == 1:
            # 1D coordinates (rectilinear grid) - need to create 2D meshgrid
            target_lon_2d, target_lat_2d = np.meshgrid(target_lon.data, target_lat.data)
            self.target_shape = target_lat_2d.shape
            target_lat_flat = target_lat_2d.flatten()
            target_lon_flat = target_lon_2d.flatten()
        else:
            # 2D coordinates (curvilinear grid) - use as is
            self.target_shape = target_lat.shape
            target_lat_flat = target_lat.data.flatten()
            target_lon_flat = target_lon.data.flatten()

        # Clamp coordinates to valid ranges to handle edge cases gracefully
        target_lat_flat = np.clip(target_lat_flat, -90.0, 90.0)
        target_lon_flat = np.clip(target_lon_flat, -180.0, 180.0)

        # Transform to 3D coordinates (assuming height=0 for surface points)
        target_heights = np.zeros_like(target_lat_flat)
        self.target_x, self.target_y, self.target_z = self.coordinate_transformer.transform_coordinates(
            target_lon_flat, target_lat_flat, target_heights
        )

        # Check for finite values before creating 3D points array
        if not (
            np.isfinite(self.target_x).all() and np.isfinite(self.target_y).all() and np.isfinite(self.target_z).all()
        ):
            # Identify problematic coordinates
            non_finite_mask = ~(np.isfinite(self.target_x) & np.isfinite(self.target_y) & np.isfinite(self.target_z))
            if np.any(non_finite_mask):
                problematic_lats = target_lat_flat[non_finite_mask]
                problematic_lons = target_lon_flat[non_finite_mask]
                raise ValueError(
                    f"Non-finite coordinates found during transformation: "
                    f"lat={problematic_lats[:5]}, lon={problematic_lons[:5]} "
                    f"(showing first 5 of {np.sum(non_finite_mask)} non-finite points)"
                )

        # Store as 3D points array
        self.target_points_3d = np.column_stack([self.target_x, self.target_y, self.target_z])

    def _build_interpolation_structures(self) -> None:
        """Build interpolation structures based on method."""
        # Create interpolation engine
        self.interpolation_engine = InterpolationEngine(
            method=self.method, spherical=self.spherical, fill_method=self.fill_method, extrapolate=self.extrapolate
        )

        # Build the interpolation structures
        self.interpolation_engine.build_structures(
            self.source_points_3d, self.target_points_3d, self.radius_of_influence
        )

    def _precompute_interpolation_weights(self) -> None:
        """Precompute interpolation weights for build-once/apply-many pattern."""
        # The interpolation engine already precomputes weights during build_structures
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
            raise TypeError("Input must be xarray DataArray or Dataset")

    def _interpolate_dataarray(self, data: xr.DataArray) -> xr.DataArray:
        """Interpolate a single DataArray."""
        # Validate that data coordinates match source grid
        if not self._validate_data_coordinates(data):
            raise ValueError("Data coordinates do not match source grid")

        # Find spatial dimensions in the data that match the source grid shape
        # The data should have the same spatial dimensions as the source grid
        source_lat_dims = self.source_grid[self.source_lat_name].dims

        # If the data has the same dimensions as the source grid coordinates, use those
        if all(dim in data.dims for dim in source_lat_dims):
            spatial_dims = source_lat_dims
        else:
            # Otherwise, find dimensions that match the source grid shape
            spatial_dims = []
            for dim in data.dims:
                if data.sizes[dim] in self.source_shape:
                    spatial_dims.append(dim)
            spatial_dims = tuple(spatial_dims[:2])  # Take first two matching dimensions that match source shape

        # If we still don't have 2 spatial dimensions, use the last two dimensions as a fallback
        if len(spatial_dims) != 2:
            spatial_dims = tuple(data.dims[-2:])  # Use last two dimensions as spatial

        # Define wrapper function for apply_ufunc
        def _apply_interpolation(data_slice, engine=self.interpolation_engine):
            # Reshape to 1D for interpolation (flatten the spatial dimensions)
            original_shape = data_slice.shape
            # Flatten all dimensions except the last N (spatial)
            # data_slice here is expected to be (..., y, x)
            # but flatten to (..., flattened_spatial)

            # The input will be (..., source_lat, source_lon)
            # We reshape to (..., source_points_flat)
            reshaped_data = data_slice.reshape(*data_slice.shape[:-2], -1)

            # Use interpolation engine
            interpolated = engine.interpolate(reshaped_data)

            # Reshape back to target grid shape
            # The output of interpolate is (..., target_points_flat)
            # We reshape to (..., target_lat, target_lon)

            # Get target shape from the engine attributes or grid properties
            target_lat = self.target_grid[self.target_lat_name]
            if target_lat.ndim == 2:
                 target_shape = target_lat.shape
            else:
                 target_shape = (self.target_grid[self.target_lat_name].size,
                                 self.target_grid[self.target_lon_name].size)

            final_shape = (*data_slice.shape[:-2], *target_shape)
            return interpolated.reshape(final_shape)

        # Use xr.apply_ufunc to handle Dask arrays lazily
        # input_core_dims: the dimensions that will be consumed (source spatial dims)
        # output_core_dims: the dimensions that will be produced (target spatial dims)

        # Determine target dims
        target_lat_coord = self.target_grid[self.target_lat_name]
        target_lon_coord = self.target_grid[self.target_lon_name]

        if target_lat_coord.ndim == 2:
             target_dims = list(target_lat_coord.dims)
             target_shape = target_lat_coord.shape
        else:
             target_dims = [target_lat_coord.dims[0], target_lon_coord.dims[0]]
             target_shape = (target_lat_coord.size, target_lon_coord.size)

        # Create output coordinates dictionary for apply_ufunc (optional but good for metadata)
        # Actually apply_ufunc handles coords if we provide output_core_dims properly

        # Create dictionary mapping target dim names to sizes for apply_ufunc
        output_sizes = {dim: size for dim, size in zip(target_dims, target_shape)}

        result = xr.apply_ufunc(
            _apply_interpolation,
            data,
            input_core_dims=[list(spatial_dims)],
            output_core_dims=[target_dims],
            exclude_dims=set(spatial_dims),  # These dimensions change size
            vectorize=False,  # Handle extra dims manually in _apply_interpolation
            dask="parallelized",  # Enable Dask parallel execution
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": output_sizes},
        )

        # Attach coordinates to result
        # Coordinates from data (non-spatial) are preserved by apply_ufunc
        # We need to add target spatial coordinates

        if target_lat_coord.ndim == 2:
             result.coords[self.target_lat_name] = target_lat_coord
             result.coords[self.target_lon_name] = target_lon_coord
        else:
             result.coords[self.target_lat_name] = target_lat_coord
             result.coords[self.target_lon_name] = target_lon_coord

        # Also ensure dimension coordinates exist
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
