"""
Optimized coordinate transformation for curvilinear regridding.

This module handles efficient 3D coordinate transformations using pyproj
with vectorized operations and caching mechanisms.

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

import hashlib
import pickle

import numpy as np
import pyproj
from scipy.spatial.distance import pdist  # type: ignore


class CoordinateTransformer:
    """Optimized coordinate transformer with caching and batch processing."""

    def __init__(self, source_crs: str = "EPSG:4979", target_crs: str = "EPSG:4978"):
        """Initialize coordinate transformer.

        Args:
            source_crs: Source coordinate reference system (default: EPSG:4979 for lat/lon/height)
            target_crs: Target coordinate reference system (default: EPSG:4978 for 3D geocentric)
        """
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._max_cache_size = 100  # Maximum number of cached transformations

    def transform_coordinates(
        self, lon: np.ndarray, lat: np.ndarray, height: np.ndarray | None = None, use_cache: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform coordinates from geographic to 3D geocentric.

        Args:
            lon: Longitude array (degrees)
            lat: Latitude array (degrees)
            height: Height array (meters, optional, defaults to 0)
            use_cache: Whether to use caching for repeated transformations

        Returns:
            Tuple of (x, y, z) coordinates in target CRS
        """
        # Flatten arrays for consistent processing
        lon_flat = np.asarray(lon).flatten()
        lat_flat = np.asarray(lat).flatten()

        if height is None:
            height_flat = np.zeros_like(lon_flat)
        else:
            height_flat = np.asarray(height).flatten()

        # Create cache key based on input coordinates
        cache_key = None
        if use_cache:
            # Use hash of coordinate values as cache key (for approximate matches)
            coords_tuple = (lon_flat.tobytes(), lat_flat.tobytes(), height_flat.tobytes())
            cache_key = hashlib.sha256(pickle.dumps(coords_tuple)).hexdigest()

            if cache_key in self._cache:
                return self._cache[cache_key]

        # Perform transformation
        x, y, z = self.transformer.transform(lon_flat, lat_flat, height_flat)

        # Reshape to match original input shape
        x = x.reshape(lon.shape)
        y = y.reshape(lon.shape)
        z = z.reshape(lon.shape)

        # Cache result if caching is enabled
        if use_cache and cache_key is not None:
            # Implement LRU-like behavior by removing oldest entries when needed
            if len(self._cache) >= self._max_cache_size:
                # Remove first item (oldest)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = (x, y, z)

        return x, y, z

    def inverse_transform_coordinates(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, use_cache: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform coordinates from 3D geocentric back to geographic.

        Args:
            x: X coordinate array in target CRS
            y: Y coordinate array in target CRS
            z: Z coordinate array in target CRS
            use_cache: Whether to use caching for repeated transformations

        Returns:
            Tuple of (lon, lat, height) coordinates in source CRS
        """
        # Flatten arrays for consistent processing
        x_flat = np.asarray(x).flatten()
        y_flat = np.asarray(y).flatten()
        z_flat = np.asarray(z).flatten()

        # Create cache key based on input coordinates
        cache_key = None
        if use_cache:
            coords_tuple = (x_flat.tobytes(), y_flat.tobytes(), z_flat.tobytes())
            cache_key = hashlib.sha256(pickle.dumps(coords_tuple)).hexdigest()

            if cache_key in self._cache:
                return self._cache[cache_key]

        # Perform inverse transformation
        lon, lat, height = self.transformer.transform(x_flat, y_flat, z_flat, direction="INVERSE")

        # Reshape to match original input shape
        lon = lon.reshape(x.shape)
        lat = lat.reshape(x.shape)
        height = height.reshape(z.shape)

        # Cache result if caching is enabled
        if use_cache and cache_key is not None:
            # Implement LRU-like behavior by removing oldest entries when needed
            if len(self._cache) >= self._max_cache_size:
                # Remove first item (oldest)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = (lon, lat, height)

        return lon, lat, height

    def calculate_distance_threshold(self, points_3d: np.ndarray, factor: float = 3.0) -> float:
        """Calculate appropriate distance threshold for out-of-domain detection.

        Args:
            points_3d: Array of 3D points (n, 3) in geocentric coordinates
            factor: Multiplier for average distance between points

        Returns:
            Distance threshold value
        """
        if len(points_3d) < 2:
            return float("inf")

        # Calculate typical inter-point distances in the source domain
        # Use a sample of points to estimate average distance
        if len(points_3d) > 100:
            # Sample 100 random points if we have many points
            indices = np.random.choice(len(points_3d), 100, replace=False)
            sample_points = points_3d[indices]
        else:
            sample_points = points_3d

        # Calculate distances between all pairs of sample points
        # For efficiency, only calculate for a subset if we have many points
        if len(sample_points) > 50:
            # Calculate distances for a random subset of pairs
            n_pairs = min(500, len(sample_points) * (len(sample_points) - 1) // 2)
            if n_pairs > 0:
                # Random sampling approach
                distances = []
                for _ in range(min(100, n_pairs)):
                    i, j = np.random.choice(len(sample_points), 2, replace=False)
                    dist = np.linalg.norm(sample_points[i] - sample_points[j])
                    distances.append(dist)
                avg_dist = float(np.mean(distances)) if distances else 0.0
            else:
                avg_dist = 0.0
        else:
            # Calculate all pairwise distances for smaller datasets
            distances = pdist(sample_points)
            avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.0

        # Set threshold as a multiple of average distance
        return avg_dist * factor

    def clear_cache(self) -> None:
        """Clear the transformation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {"size": len(self._cache), "max_size": self._max_cache_size}


# Pre-configured transformer instance for common use cases
WGS84_TO_3D = CoordinateTransformer("EPSG:4979", "EPSG:4978")
