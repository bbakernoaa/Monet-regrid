"""
Core regridder classes for monet-regrid.

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
import json
from collections.abc import Hashable
from typing import Any

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr


from monet_regrid.curvilinear import CurvilinearInterpolator
from monet_regrid.methods import conservative, interp
from monet_regrid.methods.flox_reduce import compute_mode, statistic_reduce
from monet_regrid.utils import (
    _create_cache_key,
    format_for_regrid,
    identify_cf_coordinates,
    validate_input,
)


class BaseRegridder(abc.ABC):
    """Abstract base class for regridder implementations.

    This class defines the interface for all regridder implementations in monet-regrid.
    It provides common functionality and ensures consistent API across different grid types.
    """

    def __init__(
        self,
        source_data: xr.DataArray | xr.Dataset | None,
        target_grid: xr.Dataset,
    ):
        """Initialize the regridder with source data and target grid.
        Args:
            source_data: The source data to be regridded, or None if the
                regridder is being loaded from a file.
            target_grid: The target grid specification as a Dataset.
        """
        self.source_data = source_data
        self.target_grid = target_grid
        self._validate_inputs()

    @abc.abstractmethod
    def __call__(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation.

        Args:
            **kwargs: Additional arguments for the regridding operation

        Returns:
            Regridded data with the same type as input (DataArray or Dataset)
        """
        pass

    @abc.abstractmethod
    def to_file(self, filepath: str) -> None:
        """Save the regridder to a file.

        Args:
            filepath: Path to save the regridder
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_file(cls, filepath: str) -> BaseRegridder:
        """Load a regridder from a file.

        Args:
            filepath: Path to load the regridder from

        Returns:
            Instance of the regridder class
        """
        pass

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Get information about the regridder instance.

        Returns:
            Dictionary containing regridder metadata and configuration
        """
        pass

    def _validate_inputs(self) -> None:
        """Validate the source data and target grid inputs."""
        if self.source_data is not None:
            if not isinstance(self.source_data, (xr.DataArray, xr.Dataset)):
                msg = "source_data must be an xarray DataArray or Dataset"
                raise TypeError(msg)
            # Defer source coordinate identification until regridding is called
            try:
                self.source_lat_name, self.source_lon_name = identify_cf_coordinates(
                    self.source_data
                )
            except ValueError as e:
                raise ValueError(f"Source data validation failed: {e}") from e

        if not isinstance(self.target_grid, xr.Dataset):
            msg = "target_grid must be an xarray Dataset"
            raise TypeError(msg)

        try:
            self.target_lat_name, self.target_lon_name = identify_cf_coordinates(
                self.target_grid
            )
        except ValueError as e:
            raise ValueError(f"Target grid validation failed: {e}") from e

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the regridder for serialization (Dask compatibility)."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the regridder from serialized state (Dask compatibility)."""
        self.__dict__.update(state)


class RectilinearRegridder(BaseRegridder):
    """Regridder implementation for rectilinear grids using interpolation methods.

    This class handles regridding between rectilinear grids using various interpolation
    methods like linear, nearest-neighbor, bilinear, cubic, and conservative approaches.
    """

    def __init__(
        self,
        source_data: xr.DataArray | xr.Dataset,
        target_grid: xr.Dataset,
        method: str = "linear",
        time_dim: str | None = "time",
        **kwargs: Any,
    ):
        """Initialize the rectilinear regridder.

        Args:
            source_data: The source data to be regridded (DataArray or Dataset)
            target_grid: The target grid specification as a Dataset
            method: Interpolation method ('linear', 'nearest', 'cubic', 'bilinear', 'conservative')
            time_dim: Name of the time dimension, or None to force regridding over time
            **kwargs: Additional method-specific arguments
        """
        self.method = method
        self.time_dim = time_dim
        self.method_kwargs = kwargs
        # Add caching for validated target grid and formatted data
        self._validation_cache: dict[tuple, xr.Dataset] = {}
        self._formatting_cache: dict[tuple, xr.DataArray | xr.Dataset] = {}
        super().__init__(source_data, target_grid)

    def __call__(self, data: xr.DataArray | xr.Dataset | None = None, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation using interpolation methods.

        Args:
            data: Data to regrid (optional, defaults to source_data from initialization)
            **kwargs: Additional arguments that override initialization parameters

        Returns:
            Regridded data with the same type as input (DataArray or Dataset)
        """
        # Use provided data or fall back to source data
        input_data = data if data is not None else self.source_data

        # Override with any runtime kwargs
        method = kwargs.get("method", self.method)
        time_dim = kwargs.get("time_dim", self.time_dim)
        method_kwargs = {**self.method_kwargs, **{k: v for k, v in kwargs.items() if k not in ["method", "time_dim"]}}

        # Create a stable cache key
        cache_key = (
            _create_cache_key(input_data, time_dim),
            _create_cache_key(self.target_grid),
        )

        # Check if we have cached validated target grid
        if cache_key in self._validation_cache:
            validated_target_grid = self._validation_cache[cache_key]
        else:
            # Validate inputs
            validated_target_grid = validate_input(input_data, self.target_grid, time_dim)
            # Cache the validated target grid
            self._validation_cache[cache_key] = validated_target_grid

        # Check if we have cached formatted data
        if cache_key in self._formatting_cache:
            formatted_data = self._formatting_cache[cache_key]
        else:
            # Format data for regridding
            formatted_data = format_for_regrid(input_data, validated_target_grid)
            # Cache the formatted data
            self._formatting_cache[cache_key] = formatted_data

        # Apply the appropriate method
        if method in ["linear", "nearest", "cubic", "bilinear"]:
            return interp.interp_regrid(formatted_data, validated_target_grid, method)
        elif method == "conservative":
            # Handle conservative regridding with its specific parameters
            latitude_coord = method_kwargs.get("latitude_coord", None)
            skipna = method_kwargs.get("skipna", True)
            nan_threshold = method_kwargs.get("nan_threshold", 1.0)
            output_chunks = method_kwargs.get("output_chunks", None)

            return conservative.conservative_regrid(
                formatted_data,
                validated_target_grid,
                latitude_coord,
                skipna,
                nan_threshold,
                output_chunks,
            )
        else:
            msg = f"Unsupported method: {method}. Supported methods are: linear, nearest, cubic, bilinear, conservative"
            raise ValueError(msg)

    def to_file(self, filepath: str) -> None:
        """Save the regridder configuration to a NetCDF file.

        Args:
            filepath: Path to save the regridder configuration
        """
        # Create a copy to avoid modifying the original object's state
        target_grid_to_save = self.target_grid.copy()

        # Serialize the configuration to a JSON string
        config = {
            "method": self.method,
            "time_dim": self.time_dim,
            "method_kwargs": self.method_kwargs,
        }
        config_json = json.dumps(config)

        # Save the target grid and configuration attributes to a NetCDF file
        target_grid_to_save.attrs["regridder_config"] = config_json
        target_grid_to_save.to_netcdf(filepath, engine="h5netcdf")

    @classmethod
    def from_file(cls, filepath: str) -> RectilinearRegridder:
        """Load a regridder from a NetCDF file.

        Args:
            filepath: Path to load the regridder from

        Returns:
            Instance of RectilinearRegridder
        """
        # Open the NetCDF file
        with xr.open_dataset(filepath, engine="h5netcdf") as ds:
            # Load the target grid and configuration
            target_grid = ds
            config_json = ds.attrs.get("regridder_config")

        if config_json is None:
            msg = "Regridder configuration not found in the file attributes."
            raise ValueError(msg)

        # Deserialize the configuration
        config = json.loads(config_json)

        # Create an instance of the class with a placeholder for source_data
        return cls(
            source_data=None,  # source_data is not stored in the file
            target_grid=target_grid,
            method=config["method"],
            time_dim=config["time_dim"],
            **config["method_kwargs"],
        )

    def info(self) -> dict[str, Any]:
        """Get information about the rectilinear regridder instance.

        Returns:
            Dictionary containing regridder metadata and configuration
        """
        source_dims = {}
        if hasattr(self.source_data, "dims"):
            # Convert dims to a dict format (name -> size)
            if hasattr(self.source_data, "sizes"):
                source_dims = {dim: self.source_data.sizes[dim] for dim in self.source_data.dims}
            else:
                source_dims = {
                    dim: len(self.source_data[dim]) if dim in self.source_data.dims else 0
                    for dim in self.source_data.dims
                }

        return {
            "type": "RectilinearRegridder",
            "method": self.method,
            "time_dim": self.time_dim,
            "method_kwargs": self.method_kwargs,
            "source_dims": source_dims,
            "target_coords": list(self.target_grid.coords),
            "grid_type": "rectilinear",
        }

    def stat(
        self,
        method: str,
        time_dim: str | None = "time",
        skipna: bool = False,
        fill_value: None | Any = None,
    ) -> xr.DataArray | xr.Dataset:
        """Upsampling of data using statistical methods (e.g. the mean or variance).

        Args:
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

        ds_formatted = format_for_regrid(self.source_data, self.target_grid, stats=True)

        return statistic_reduce(ds_formatted, self.target_grid, time_dim, method, skipna, fill_value)

    def most_common(
        self,
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
        if isinstance(self.source_data, xr.Dataset):
            msg = (
                "The 'most common value' regridder is not implemented for\n",
                "xarray.Dataset, as it requires specifying the expected labels.\n"
                "Please select only a single variable (as DataArray),\n"
                " and regrid it separately.",
            )
            raise ValueError(msg)

        ds_formatted = format_for_regrid(self.source_data, self.target_grid, stats=True)

        return compute_mode(
            ds_formatted,
            self.target_grid,
            values,
            time_dim,
            fill_value,
            anti_mode=False,
        )

    def least_common(
        self,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray:
        """Regrid by taking the least common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "least common" one will randomly be
        either of the two.

        Args:
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
        if isinstance(self.source_data, xr.Dataset):
            msg = (
                "The 'least common value' regridder is not implemented for\n",
                "xarray.Dataset, as it requires specifying the expected labels.\n"
                "Please select only a single variable (as DataArray),\n"
                " and regrid it separately.",
            )
            raise ValueError(msg)

        ds_formatted = format_for_regrid(self.source_data, self.target_grid, stats=True)

        return compute_mode(
            ds_formatted,
            self.target_grid,
            values,
            time_dim,
            fill_value,
            anti_mode=True,
        )


class CurvilinearRegridder(BaseRegridder):
    """Regridder implementation for curvilinear grids using 3D coordinate transformations.

    This class handles regridding between curvilinear grids using the CurvilinearInterpolator
    which performs interpolation in 3D geocentric coordinates for accurate spherical geometry.
    """

    def __init__(
        self, source_data: xr.DataArray | xr.Dataset, target_grid: xr.Dataset, method: str = "linear", **kwargs: Any
    ):
        """Initialize the curvilinear regridder.

        Args:
            source_data: The source data to be regridded (DataArray or Dataset)
            target_grid: The target grid specification as a Dataset
            method: Interpolation method for curvilinear grids
            **kwargs: Additional method-specific arguments
        """
        self.method = method
        self.method_kwargs = kwargs
        super().__init__(source_data, target_grid)

    def __call__(self, data: xr.DataArray | xr.Dataset | None = None, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation for curvilinear grids.
        Args:
            data: Data to regrid (optional, defaults to source_data from initialization)
            **kwargs: Additional arguments that override initialization parameters
        Returns:
            Regridded data with the same type as input (DataArray or Dataset)
        """
        # Use provided data or fall back to source data
        input_data = data if data is not None else self.source_data
        if input_data is None:
            raise ValueError(
                "Regridding requires data to be provided, either at initialization or during call."
            )

        # Identify source coordinates if they haven't been already (e.g., when loaded from file)
        if not hasattr(self, "source_lat_name"):
            try:
                self.source_lat_name, self.source_lon_name = identify_cf_coordinates(
                    input_data
                )
            except ValueError as e:
                raise ValueError(f"Source data validation failed during call: {e}") from e

        # Override with any runtime kwargs
        method = kwargs.get("method", self.method)
        method_kwargs = {
            **self.method_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ["method"]},
        }

        # Create the CurvilinearInterpolator
        source_grid = self._create_source_grid_from_data(input_data)
        # Create the interpolator with the source and target grids
        interpolator = CurvilinearInterpolator(
            source_grid=source_grid,
            target_grid=self.target_grid,
            source_lat_name=self.source_lat_name,
            source_lon_name=self.source_lon_name,
            target_lat_name=self.target_lat_name,
            target_lon_name=self.target_lon_name,
            method=method,
            **method_kwargs,
        )

        # Apply the interpolation to the actual data
        result = interpolator(input_data)

        return result

    def _create_source_grid_from_data(self, source_data: xr.DataArray | xr.Dataset | None = None) -> xr.Dataset:
        """Create a grid specification from source data."""
        # Use provided data or fall back to source data
        data = source_data if source_data is not None else self.source_data

        # Extract coordinate information from source data
        # First, determine the coordinate names using cf-xarray if available
        try:
            lat_coord = data.cf["latitude"]
            lon_coord = data.cf["longitude"]
            lat_name = lat_coord.name
            lon_name = lon_coord.name

            # Extract the coordinate variables
            source_grid = xr.Dataset({lat_name: data[lat_name], lon_name: data[lon_name]})

            return source_grid
        except (KeyError, AttributeError):
            # Fallback to manual search
            lat_coords = [name for name in data.coords if "lat" in str(name).lower() or "latitude" in str(name).lower()]
            lon_coords = [
                name for name in data.coords if "lon" in str(name).lower() or "longitude" in str(name).lower()
            ]

            if lat_coords and lon_coords:
                # If lat/lon coordinates are found in the data, use them
                lat_name = lat_coords[0]
                lon_name = lon_coords[0]

                source_grid = xr.Dataset({lat_name: data[lat_name], lon_name: data[lon_name]})

                return source_grid
            # If no explicit lat/lon coordinates are found in the data,
            # we need to infer the spatial dimensions from the data shape
            # and use the source grid that was provided during initialization
            # In this case, the CurvilinearInterpolator should be initialized differently
            # This is a complex scenario - for now, let's assume that the source grid
            # coordinates were already provided during initialization and we can
            # extract spatial coordinate information from the data dimensions
            # by assuming the last two dimensions are spatial
            elif len(data.dims) >= 2:
                # Use the last two dimensions as spatial dimensions
                y_dim, x_dim = data.dims[-2], data.dims[-1]

                # Create simple coordinate arrays based on the spatial dimensions
                y_coords = np.arange(data.sizes[y_dim])
                x_coords = np.arange(data.sizes[x_dim])

                # Create 2D coordinate grids
                lon_2d, lat_2d = np.meshgrid(x_coords, y_coords)

                # Create a simple coordinate dataset
                source_grid = xr.Dataset({"latitude": (["y", "x"], lat_2d), "longitude": (["y", "x"], lon_2d)})

                return source_grid
            else:
                msg = "Source data must have at least 2 dimensions for curvilinear regridding"
                raise ValueError(msg)


    def to_file(self, filepath: str) -> None:
        """Save the regridder configuration to a NetCDF file.

        Args:
            filepath: Path to save the regridder configuration
        """
        # Create a copy to avoid modifying the original object's state
        target_grid_to_save = self.target_grid.copy()

        # Serialize the configuration to a JSON string
        config = {"method": self.method, "method_kwargs": self.method_kwargs}
        config_json = json.dumps(config)

        # Save the target grid and configuration attributes to a NetCDF file
        target_grid_to_save.attrs["regridder_config"] = config_json
        target_grid_to_save.to_netcdf(filepath, engine="h5netcdf")

    @classmethod
    def from_file(cls, filepath: str) -> CurvilinearRegridder:
        """Load a regridder from a NetCDF file.

        Args:
            filepath: Path to load the regridder from

        Returns:
            Instance of CurvilinearRegridder
        """
        # Open the NetCDF file
        with xr.open_dataset(filepath, engine="h5netcdf") as ds:
            # Load the target grid and configuration
            target_grid = ds
            config_json = ds.attrs.get("regridder_config")

        if config_json is None:
            msg = "Regridder configuration not found in the file attributes."
            raise ValueError(msg)

        # Deserialize the configuration
        config = json.loads(config_json)

        # Create an instance of the class with a placeholder for source_data
        return cls(
            source_data=None,  # source_data is not stored in the file
            target_grid=target_grid,
            method=config["method"],
            **config["method_kwargs"],
        )

    def info(self) -> dict[str, Any]:
        """Get information about the curvilinear regridder instance.

        Returns:
            Dictionary containing regridder metadata and configuration
        """
        source_dims = {}
        if hasattr(self.source_data, "dims"):
            # Convert dims to a dict format (name -> size)
            if hasattr(self.source_data, "sizes"):
                source_dims = {dim: self.source_data.sizes[dim] for dim in self.source_data.dims}
            else:
                source_dims = {
                    dim: len(self.source_data[dim]) if dim in self.source_data.dims else 0
                    for dim in self.source_data.dims
                }

        return {
            "type": "CurvilinearRegridder",
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "source_dims": source_dims,
            "target_coords": list(self.target_grid.coords),
            "grid_type": "curvilinear",
            "status": "implemented",
        }
