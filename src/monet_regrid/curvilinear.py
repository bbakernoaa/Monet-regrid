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

import logging
from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, cKDTree, qhull

from monet_regrid.coordinate_transformer import CoordinateTransformer
from monet_regrid.interpolation import InterpolationEngine
from monet_regrid.interpolation.utils import (
    _compute_barycentric_weights_3d,
    _point_in_tetrahedron,
)


def _apply_interpolation_wrapper(data_slice, engine, target_shape):
    """Wrapper for interpolation to be used with apply_ufunc (picklable)."""
    # Reshape to 1D for interpolation (flatten the spatial dimensions)
    # The input will be (..., source_lat, source_lon)
    # We reshape to (..., source_points_flat)
    reshaped_data = data_slice.reshape(*data_slice.shape[:-2], -1)

    # Use interpolation engine
    interpolated = engine.interpolate(reshaped_data)

    # Reshape back to target grid shape
    # The output of interpolate is (..., target_points_flat)
    # We reshape to (..., target_lat, target_lon)
    final_shape = (*data_slice.shape[:-2], *target_shape)
    return interpolated.reshape(final_shape)


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
        source_lat_name: str | None = None,
        source_lon_name: str | None = None,
        target_lat_name: str | None = None,
        target_lon_name: str | None = None,
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
                self._triangle_centroids = np.mean(self.source_points_3d[simplices], axis=1)
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
        if (
            hasattr(self.interpolation_engine, "precomputed_weights")
            and self.interpolation_engine.precomputed_weights is not None
        ):
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
        source_lat_flat = np.clip(source_lat_flat, -90.0, 90.0)  # type: ignore[assignment]
        source_lon_flat = np.clip(source_lon_flat, -180.0, 180.0)  # type: ignore[assignment]

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
                msg = (
                    f"Non-finite coordinates found during transformation: "
                    f"lat={problematic_lats[:5]}, lon={problematic_lons[:5]} "
                    f"(showing first 5 of {np.sum(non_finite_mask)} non-finite points)"
                )
                raise ValueError(
                    msg
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
        target_lat_flat = np.clip(target_lat_flat, -90.0, 90.0)  # type: ignore[assignment]
        target_lon_flat = np.clip(target_lon_flat, -180.0, 180.0)  # type: ignore[assignment]

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
                msg = (
                    f"Non-finite coordinates found during transformation: "
                    f"lat={problematic_lats[:5]}, lon={problematic_lons[:5]} "
                    f"(showing first 5 of {np.sum(non_finite_mask)} non-finite points)"
                )
                raise ValueError(
                    msg
                )

        # Store as 3D points array
        self.target_points_3d = np.column_stack([self.target_x, self.target_y, self.target_z])

    def _build_interpolation_structures(self) -> None:
        """Build interpolation structures based on method."""
        # Create interpolation engine
        self.interpolation_engine = InterpolationEngine(
            method=self.method, spherical=self.spherical, fill_method=self.fill_method, extrapolate=self.extrapolate
        )

        if self.method == "conservative":
            # Extract boundaries (code omitted for brevity, same as previous)
            # Helper to get bounds
            def get_bounds(ds, lat_name, lon_name):
                # Try to find bounds attribute
                try:
                    lat_bounds_name = ds[lat_name].attrs.get("bounds", f"{lat_name}_bnds")
                    lon_bounds_name = ds[lon_name].attrs.get("bounds", f"{lon_name}_bnds")

                    if lat_bounds_name in ds and lon_bounds_name in ds:
                        # Reshape to (N, 4, 2) format
                        lat_b = ds[lat_bounds_name].values
                        lon_b = ds[lon_bounds_name].values

                        if lat_b.ndim == 3 and lat_b.shape[-1] == 4:
                            n_cells = lat_b.shape[0] * lat_b.shape[1]
                            lat_b_flat = lat_b.reshape(n_cells, 4)
                            lon_b_flat = lon_b.reshape(n_cells, 4)
                            return np.stack([lon_b_flat, lat_b_flat], axis=2)
                except qhull.QhullError as e:
                    logging.warning(f"Could not create triangulation, falling back to nearest neighbor: {e}")

                msg = (
                    f"Conservative regridding requires explicit bounds for {lat_name} and {lon_name}. "
                    "Please ensure 'bounds' attribute is set and variables exist with shape (y, x, 4)."
                )
                raise ValueError(
                    msg
                )

            source_vertices = get_bounds(self.source_grid, self.source_lat_name, self.source_lon_name)
            target_vertices = get_bounds(self.target_grid, self.target_lat_name, self.target_lon_name)

            self.interpolation_engine.build_conservative_structures(
                self.source_points_3d,
                self.target_points_3d,
                source_vertices,
                target_vertices,
                radius_of_influence=self.radius_of_influence,
            )
        elif self.method in ["bilinear", "cubic"]:
            # Structured interpolation requires source shape
            self.interpolation_engine.build_structures(
                self.source_points_3d,
                self.target_points_3d,
                self.radius_of_influence,
                source_shape=self.source_shape,  # type: ignore[arg-type]
            )
        else:
            # Standard interpolation
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
            msg = "Input must be xarray DataArray or Dataset"
            raise TypeError(msg)

    def _interpolate_dataarray(self, data: xr.DataArray) -> xr.DataArray:
        """Interpolate a single DataArray."""
        # Validate that data coordinates match source grid
        if not self._validate_data_coordinates(data):
            msg = "Data coordinates do not match source grid"
            raise ValueError(msg)

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

        # Determine target dims and shape
        target_lat_coord = self.target_grid[self.target_lat_name]
        target_lon_coord = self.target_grid[self.target_lon_name]

        if target_lat_coord.ndim == 2:
            target_dims = list(target_lat_coord.dims)
            target_shape = target_lat_coord.shape
        else:
            target_dims = [target_lat_coord.dims[0], target_lon_coord.dims[0]]
            target_shape = (target_lat_coord.size, target_lon_coord.size)

        # Create dictionary mapping target dim names to sizes for apply_ufunc
        output_sizes = dict(zip(target_dims, target_shape, strict=False))

        # Use xr.apply_ufunc to handle Dask arrays lazily
        result = xr.apply_ufunc(
            _apply_interpolation_wrapper,
            data,
            kwargs={"engine": self.interpolation_engine, "target_shape": target_shape},
            input_core_dims=[list(spatial_dims)],
            output_core_dims=[target_dims],
            exclude_dims=set(spatial_dims),  # These dimensions change size
            vectorize=False,  # Handle extra dims manually in _apply_interpolation
            dask="parallelized",  # Enable Dask parallel execution
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": output_sizes},
            keep_attrs=True,
        )

        # Manually ensure attributes are preserved if apply_ufunc didn't do it
        if not result.attrs and data.attrs:
            result.attrs = data.attrs.copy()

        # DEBUG
        if not result.attrs:
            pass

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

        # DEBUG
        if not result.attrs:
            pass

        return result  # type: ignore[no-any-return]

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
