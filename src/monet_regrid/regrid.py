from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

"""
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
optimizations, adding curvilinear support, URLs updated,
and documentation adapted for new branding.
"""

from monet_regrid.constants import GridType
from monet_regrid.core import BaseRegridder, CurvilinearRegridder, RectilinearRegridder
from monet_regrid.utils import (
    _get_grid_type,
    validate_input,
)


@xr.register_dataarray_accessor("regrid")
@xr.register_dataset_accessor("regrid")
class Regridder:
    """Regridding xarray datasets and dataarrays.

    Available methods:
        linear: linear, bilinear, or higher dimensional linear interpolation
        nearest: nearest-neighbor regridding
        cubic: cubic spline regridding
        conservative: conservative regridding
        most_common: most common value regridder
        stat: area statistics regridder
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj

    def build_regridder(self, ds_target_grid: xr.Dataset, method: str = "linear", **kwargs: Any) -> BaseRegridder:
        """Factory method to build the appropriate regridder based on grid type.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            method: The regridding method to use (e.g., 'linear', 'nearest', 'cubic', 'conservative').
            **kwargs: Additional keyword arguments to pass to the regridder.

        Returns:
            An instance of the appropriate regridder class based on grid type.
        """
        # Detect grid types for both source and target

        # For grid type detection, we need to pass the dataset that contains the coordinate variables
        # If self._obj is a Dataset with coordinate variables like RASM (xc, yc), use it directly
        # If self._obj is a DataArray, convert it to Dataset first to access all coordinates
        try:
            # Convert to Dataset if needed to ensure we have access to all coordinate information
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # Convert to dataset to include all coordinates
                temp_ds = self._obj.to_dataset()
                source_grid_type = _get_grid_type(temp_ds)
        except (KeyError, AttributeError, ValueError):
            # If we can't identify coordinates in the main object, try to create a dataset with just the coordinates
            try:
                coord_vars = {}
                for coord_name in self._obj.coords:
                    coord_var = self._obj.coords[coord_name]
                    # Include coordinate variables that might represent spatial coordinates
                    if coord_var.ndim >= 1 and any(
                        keyword in str(coord_name).lower() for keyword in ["lat", "lon", "x", "y", "xc", "yc"]
                    ):
                        coord_vars[coord_name] = coord_var

                if coord_vars:
                    temp_ds = xr.Dataset(coord_vars)
                    source_grid_type = _get_grid_type(temp_ds)
                else:
                    source_grid_type = GridType.RECTILINEAR
            except (KeyError, AttributeError, ValueError):
                # If all attempts fail, default to rectilinear
                source_grid_type = GridType.RECTILINEAR

        target_grid_type = _get_grid_type(ds_target_grid)

        # Choose the appropriate regridder based on grid types
        if GridType.CURVILINEAR in (source_grid_type, target_grid_type):
            # Use CurvilinearRegridder for any curvilinear grid scenario

            return CurvilinearRegridder(source_data=self._obj, target_grid=ds_target_grid, method=method, **kwargs)
        else:
            # Use RectilinearRegridder for rectilinear-to-rectilinear regridding
            return RectilinearRegridder(source_data=self._obj, target_grid=ds_target_grid, method=method, **kwargs)

    def linear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with linear interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # For DataArray, we need to check its coordinates to detect grid type
                # Create a temporary dataset with just the coordinate variables
                coord_vars = {}
                for coord_name in self._obj.coords:
                    coord_var = self._obj.coords[coord_name]
                    # Include coordinate variables that might represent spatial coordinates
                    if coord_var.ndim >= 1 and any(
                        keyword in str(coord_name).lower() for keyword in ["lat", "lon", "x", "y", "xc", "yc"]
                    ):
                        coord_vars[coord_name] = coord_var

                if coord_vars:
                    temp_ds = xr.Dataset(coord_vars)
                    source_grid_type = _get_grid_type(temp_ds)
                else:
                    source_grid_type = GridType.RECTILINEAR
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            # We should still validate that coordinates exist but skip the dimension check
            ds_target_grid = ds_target_grid

        regridder = self.build_regridder(ds_target_grid=ds_target_grid, method="linear", time_dim=time_dim)
        return regridder()

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target with nearest-neighbor interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # For DataArray, we need to check its coordinates to detect grid type
                # Create a temporary dataset with just the coordinate variables
                coord_vars = {}
                for coord_name in self._obj.coords:
                    coord_var = self._obj.coords[coord_name]
                    # Include coordinate variables that might represent spatial coordinates
                    if coord_var.ndim >= 1 and any(
                        keyword in str(coord_name).lower() for keyword in ["lat", "lon", "x", "y", "xc", "yc"]
                    ):
                        coord_vars[coord_name] = coord_var

                if coord_vars:
                    temp_ds = xr.Dataset(coord_vars)
                    source_grid_type = _get_grid_type(temp_ds)
                else:
                    source_grid_type = GridType.RECTILINEAR
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        regridder = self.build_regridder(ds_target_grid=ds_target_grid, method="nearest", time_dim=time_dim)
        return regridder()

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with cubic interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # For DataArray, we need to check its coordinates to detect grid type
                # Create a temporary dataset with just the coordinate variables
                coord_vars = {}
                for coord_name in self._obj.coords:
                    coord_var = self._obj.coords[coord_name]
                    # Include coordinate variables that might represent spatial coordinates
                    if coord_var.ndim >= 1 and any(
                        keyword in str(coord_name).lower() for keyword in ["lat", "lon", "x", "y", "xc", "yc"]
                    ):
                        coord_vars[coord_name] = coord_var

                if coord_vars:
                    temp_ds = xr.Dataset(coord_vars)
                    source_grid_type = _get_grid_type(temp_ds)
                else:
                    source_grid_type = GridType.RECTILINEAR
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        regridder = self.build_regridder(ds_target_grid=ds_target_grid, method="cubic", time_dim=time_dim)
        return regridder()

    def conservative(
        self,
        ds_target_grid: xr.Dataset,
        latitude_coord: str | None = None,
        time_dim: str | None = "time",
        skipna: bool = True,
        nan_threshold: float = 1.0,
        output_chunks: dict[Hashable, int] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with a conservative scheme.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            latitude_coord: Name of the latitude coord, to be used for applying the
                spherical correction. By default, attempt to infer a latitude coordinate
                as either "latitude" or "lat".
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            skipna: If True, enable handling for NaN values. This adds only a small
                amount of overhead, but can be disabled for optimal performance on data
                without any NaNs.
            nan_threshold: Threshold value that will retain any output points
                containing at least this many non-null input points. The default value
                is 1.0, which will keep output points containing any non-null inputs,
                while a value of 0.0 will only keep output points where all inputs are
                non-null.
            output_chunks: Optional dictionary of explicit chunk sizes for the output
                data. If not provided, the output will be chunked the same as the input
                data.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        if not 0.0 <= nan_threshold <= 1.0:
            msg = "nan_threshold must be between [0, 1]]"
            raise ValueError(msg)

        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # For DataArray, we need to check its coordinates to detect grid type
                # Create a temporary dataset with just the coordinate variables
                coord_vars = {}
                for coord_name in self._obj.coords:
                    coord_var = self._obj.coords[coord_name]
                    # Include coordinate variables that might represent spatial coordinates
                    if coord_var.ndim >= 1 and any(
                        keyword in str(coord_name).lower() for keyword in ["lat", "lon", "x", "y", "xc", "yc"]
                    ):
                        coord_vars[coord_name] = coord_var

                if coord_vars:
                    temp_ds = xr.Dataset(coord_vars)
                    source_grid_type = _get_grid_type(temp_ds)
                else:
                    source_grid_type = GridType.RECTILINEAR
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        regridder = self.build_regridder(
            ds_target_grid=ds_target_grid,
            method="conservative",
            time_dim=time_dim,
            latitude_coord=latitude_coord,
            skipna=skipna,
            nan_threshold=nan_threshold,
            output_chunks=output_chunks,
        )
        return regridder()

    def most_common(
        self,
        ds_target_grid: xr.Dataset,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray:
        """Regrid by taking the most common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "most common" one will randomly be
        either of the two.

        Args:
            ds_target_grid: Target grid dataset
            values: Numpy array containing all labels expected to be in the
                input data. For example, `np.array([0, 2, 4])`, if the data only
                contains the values 0, 2 and 4.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            fill_value: What value to fill uncovered parts of the target grid.
                By default this will be NaN, and integer type data will be cast to
                float to accomodate this.

        Returns:
            Regridded data.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # Convert to dataset to include all coordinates
                temp_ds = self._obj.to_dataset()
                source_grid_type = _get_grid_type(temp_ds)
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        # For most_common, we need to use the RectilinearRegridder directly since it has special handling
        rectilinear_regridder = RectilinearRegridder(
            source_data=self._obj, target_grid=ds_target_grid, method="most_common", time_dim=time_dim
        )
        return rectilinear_regridder.most_common(values, time_dim, fill_value)

    def least_common(
        self,
        ds_target_grid: xr.Dataset,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
    ) -> xr.DataArray:
        """Regrid by taking the least common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "least common" one will randomly be
        either of the two.

        Args:
            ds_target_grid: Target grid dataset
            values: Numpy array containing all labels expected to be in the
                input data. For example, `np.array([0, 2, 4])`, if the data only
                contains the values 0, 2 and 4.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            fill_value: What value to fill uncovered parts of the target grid.
                By default this will be NaN, and integer type data will be cast to
                float to accomodate this.

        Returns:
            Regridded data.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # Convert to dataset to include all coordinates
                temp_ds = self._obj.to_dataset()
                source_grid_type = _get_grid_type(temp_ds)
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        rectilinear_regridder = RectilinearRegridder(
            source_data=self._obj, target_grid=ds_target_grid, method="least_common", time_dim=time_dim
        )
        return rectilinear_regridder.least_common(values, time_dim, fill_value)

    def stat(
        self,
        ds_target_grid: xr.Dataset,
        method: str,
        time_dim: str | None = "time",
        skipna: bool = False,
        fill_value: None | Any = None,
    ) -> xr.DataArray | xr.Dataset:
        """Upsampling of data using statistical methods (e.g. the mean or variance).

        We use flox Aggregations to perform a "groupby" over multiple dimensions, which
        we reduce using the specified method.
        https://flox.readthedocs.io/en/latest/aggregations.html

        Args:
            ds_target_grid: Target grid dataset
            method: One of the following reduction methods: "sum", "mean", "var", "std",
                "median", "min", or "max".
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            skipna: If NaN values should be ignored.
            fill_value: What value to fill uncovered parts of the target grid.
                By default this will be NaN, and integer type data will be cast to
                float to accomodate this.

        Returns:
            xarray.dataset with regridded land cover categorical data.
        """
        # Detect grid types to determine if we need special handling for curvilinear grids

        # For curvilinear grids, we might have different dimension names, so skip validation
        try:
            # Use the same approach as build_regridder for consistency
            if isinstance(self._obj, xr.Dataset):
                source_grid_type = _get_grid_type(self._obj)
            else:  # DataArray
                # Convert to dataset to include all coordinates
                temp_ds = self._obj.to_dataset()
                source_grid_type = _get_grid_type(temp_ds)
        except (KeyError, AttributeError, ValueError):
            source_grid_type = GridType.RECTILINEAR

        try:
            target_grid_type = _get_grid_type(ds_target_grid)
        except (KeyError, AttributeError, ValueError):
            target_grid_type = GridType.RECTILINEAR

        # Only validate input if neither grid is curvilinear (to avoid dimension name mismatches)
        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            # For curvilinear grids, just ensure both have latitude/longitude coordinates
            # The actual validation will be handled by the regridder's internal validation
            ds_target_grid = ds_target_grid

        rectilinear_regridder = RectilinearRegridder(
            source_data=self._obj, target_grid=ds_target_grid, method="stat", time_dim=time_dim
        )
        return rectilinear_regridder.stat(method, time_dim, skipna, fill_value)


