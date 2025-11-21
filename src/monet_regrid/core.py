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
from typing import Any, Hashable

import numpy as np
import xarray as xr


class BaseRegridder(abc.ABC):
    """Abstract base class for regridder implementations.

    This class defines the interface for all regridder implementations in monet-regrid.
    It provides common functionality and ensures consistent API across different grid types.
    """

    def __init__(self, source_data: xr.DataArray | xr.Dataset, target_grid: xr.Dataset, **kwargs: Any):
        """Initialize the regridder with source data and target grid.

        Args:
            source_data: The source data to be regridded (DataArray or Dataset)
            target_grid: The target grid specification as a Dataset
            **kwargs: Additional keyword arguments for specific regridder implementations
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
    def to_file(self, filepath: str, **kwargs: Any) -> None:
        """Save the regridder to a file.

        Args:
            filepath: Path to save the regridder
            **kwargs: Additional arguments for file saving
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_file(cls, filepath: str, **kwargs: Any) -> BaseRegridder:
        """Load a regridder from a file.

        Args:
            filepath: Path to load the regridder from
            **kwargs: Additional arguments for file loading

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
        if not isinstance(self.source_data, (xr.DataArray, xr.Dataset)):
            raise TypeError("source_data must be an xarray DataArray or Dataset")

        if not isinstance(self.target_grid, xr.Dataset):
            raise TypeError("target_grid must be an xarray Dataset")

        # Use coordinate identification to check for latitude/longitude coordinates
        # rather than requiring exact coordinate name matches
        source_lat_coords = self._identify_lat_coords(self.source_data)
        source_lon_coords = self._identify_lon_coords(self.source_data)
        target_lat_coords = self._identify_lat_coords(self.target_grid)
        target_lon_coords = self._identify_lon_coords(self.target_grid)

        # Validate that both source and target have latitude and longitude coordinates
        if not source_lat_coords or not source_lon_coords:
            # Also check if the dimensions themselves might be latitude/longitude
            if hasattr(self.source_data, "dims"):
                # Use a more efficient approach to find lat/lon dimensions
                lat_dim = None
                lon_dim = None
                for dim in self.source_data.dims:
                    dim_lower = str(dim).lower()
                    if "lat" in dim_lower or "latitude" in dim_lower:
                        lat_dim = dim
                        break
                for dim in self.source_data.dims:
                    dim_lower = str(dim).lower()
                    if "lon" in dim_lower or "longitude" in dim_lower:
                        lon_dim = dim
                        break

                if lat_dim and lon_dim:
                    # If dimensions suggest lat/lon, that's sufficient
                    source_lat_coords = [lat_dim]
                    source_lon_coords = [lon_dim]
                else:
                    raise ValueError(
                        f"Source data must have latitude and longitude coordinates.\n"
                        f"Source coordinates: {list(self.source_data.coords) if hasattr(self.source_data, 'coords') else []}\n"
                        f"Source dimensions: {list(self.source_data.dims) if hasattr(self.source_data, 'dims') else []}"
                    )
            else:
                raise ValueError(
                    f"Source data must have latitude and longitude coordinates.\n"
                    f"Source coordinates: {list(self.source_data.coords) if hasattr(self.source_data, 'coords') else []}"
                )

        if not target_lat_coords or not target_lon_coords:
            # Also check if the dimensions themselves might be latitude/longitude
            # Use a more efficient approach to find lat/lon dimensions
            target_lat_dim = None
            target_lon_dim = None
            for dim in self.target_grid.dims:
                dim_lower = str(dim).lower()
                if "lat" in dim_lower or "latitude" in dim_lower:
                    target_lat_dim = dim
                    break
            for dim in self.target_grid.dims:
                dim_lower = str(dim).lower()
                if "lon" in dim_lower or "longitude" in dim_lower:
                    target_lon_dim = dim
                    break

            if target_lat_dim and target_lon_dim:
                # If dimensions suggest lat/lon, that's sufficient
                target_lat_coords = [target_lat_dim]
                target_lon_coords = [target_lon_dim]
            else:
                raise ValueError(
                    f"Target grid must have latitude and longitude coordinates.\n"
                    f"Target coordinates: {list(self.target_grid.coords)}\n"
                    f"Target dimensions: {list(self.target_grid.dims)}"
                )

    def _identify_lat_coords(self, data: xr.DataArray | xr.Dataset) -> list[Hashable]:
        """Identify latitude coordinates in the data using cf-xarray or name matching."""
        # First, check for cf-xarray coordinates if available
        try:
            import cf_xarray  # noqa: F401
            # Use cf-xarray to identify latitude coordinates if available
            if hasattr(data, "cf") and "latitude" in data.cf:
                return [data.cf["latitude"].name]
        except (KeyError, AttributeError):
            pass

        # Check coordinates first - include common ocean/land model coordinate names
        # Use a more efficient approach by checking for the first match
        for name in data.coords:
            name_lower = str(name).lower()
            if any(keyword in name_lower for keyword in ["lat", "latitude", "yc", "y"]):
                return [name]

        # If no coordinates found, check dimensions - include common ocean/land model dimension names
        if hasattr(data, "dims"):
            for dim in data.dims:
                dim_lower = str(dim).lower()
                if any(keyword in dim_lower for keyword in ["lat", "latitude", "yc", "y"]):
                    return [dim]

        return []

    def _identify_lon_coords(self, data: xr.DataArray | xr.Dataset) -> list[Hashable]:
        """Identify longitude coordinates in the data using cf-xarray or name matching."""
        # First, check for cf-xarray coordinates if available
        try:
            import cf_xarray  # noqa: F401
            # Use cf-xarray to identify longitude coordinates if available
            if hasattr(data, "cf") and "longitude" in data.cf:
                return [data.cf["longitude"].name]
        except (KeyError, AttributeError):
            pass

        # Check coordinates first - include common ocean/land model coordinate names
        # Use a more efficient approach by checking for the first match
        for name in data.coords:
            name_lower = str(name).lower()
            if any(keyword in name_lower for keyword in ["lon", "longitude", "xc", "x"]):
                return [name]

        # If no coordinates found, check dimensions - include common ocean/land model dimension names
        if hasattr(data, "dims"):
            for dim in data.dims:
                dim_lower = str(dim).lower()
                if any(keyword in dim_lower for keyword in ["lon", "longitude", "xc", "x"]):
                    return [dim]

        return []

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
    methods like linear, nearest-neighbor, cubic, and conservative approaches.
    """

    def __init__(
        self,
        source_data: xr.DataArray | xr.Dataset,
        target_grid: xr.Dataset,
        method: str = "linear",
        time_dim: str | None = "time",
        **kwargs: Any
    ):
        """Initialize the rectilinear regridder.

        Args:
            source_data: The source data to be regridded (DataArray or Dataset)
            target_grid: The target grid specification as a Dataset
            method: Interpolation method ('linear', 'nearest', 'cubic', 'conservative')
            time_dim: Name of the time dimension, or None to force regridding over time
            **kwargs: Additional method-specific arguments
        """
        self.method = method
        self.time_dim = time_dim
        self.method_kwargs = kwargs
        # Add caching for validated target grid and formatted data
        self._validation_cache: dict[tuple[int, str | None], xr.Dataset] = {}
        self._formatting_cache: dict[tuple[int, int], xr.DataArray | xr.Dataset] = {}
        super().__init__(source_data, target_grid, **kwargs)

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
        method_kwargs = {**self.method_kwargs, **{k: v for k, v in kwargs.items()
                                                if k not in ["method", "time_dim"]}}

        # Import here to avoid circular imports
        from monet_regrid.methods import interp, conservative
        from monet_regrid.utils import format_for_regrid
        from monet_regrid.regrid import validate_input

        # Create a cache key based on input data and time_dim
        cache_key = (id(input_data), time_dim)

        # Check if we have cached validated target grid
        if cache_key in self._validation_cache:
            validated_target_grid = self._validation_cache[cache_key]
        else:
            # Validate inputs
            validated_target_grid = validate_input(input_data, self.target_grid, time_dim)
            # Cache the validated target grid
            self._validation_cache[cache_key] = validated_target_grid

        # Check if we have cached formatted data
        format_cache_key = (id(input_data), id(validated_target_grid))
        if format_cache_key in self._formatting_cache:
            formatted_data = self._formatting_cache[format_cache_key]
        else:
            # Format data for regridding
            formatted_data = format_for_regrid(input_data, validated_target_grid)
            # Cache the formatted data
            self._formatting_cache[format_cache_key] = formatted_data

        # Apply the appropriate method
        if method in ["linear", "nearest", "cubic"]:
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
            raise ValueError(f"Unsupported method: {method}. Supported methods are: "
                           f"linear, nearest, cubic, conservative")

    def to_file(self, filepath: str, **kwargs: Any) -> None:
        """Save the regridder configuration to a file.

        Args:
            filepath: Path to save the regridder configuration
            **kwargs: Additional arguments for file saving
        """
        import pickle

        # Create a serializable representation
        config = {
            "method": self.method,
            "time_dim": self.time_dim,
            "method_kwargs": self.method_kwargs,
            "source_data": self.source_data,  # This may need special handling for Dask
            "target_grid": self.target_grid
        }

        with open(filepath, "wb") as f:
            pickle.dump(config, f)

    @classmethod
    def from_file(cls, filepath: str, **kwargs: Any) -> RectilinearRegridder:
        """Load a regridder from a file.

        Args:
            filepath: Path to load the regridder from
            **kwargs: Additional arguments for file loading

        Returns:
            Instance of RectilinearRegridder
        """
        import pickle

        with open(filepath, "rb") as f:
            config = pickle.load(f)

        return cls(
            source_data=config["source_data"],
            target_grid=config["target_grid"],
            method=config["method"],
            time_dim=config["time_dim"],
            **config["method_kwargs"]
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
                source_dims = {dim: int(len(self.source_data[dim])) if dim in self.source_data.dims else 0
                              for dim in self.source_data.dims}

        return {
            "type": "RectilinearRegridder",
            "method": self.method,
            "time_dim": self.time_dim,
            "method_kwargs": self.method_kwargs,
            "source_dims": source_dims,
            "target_coords": list(self.target_grid.coords),
            "grid_type": "rectilinear"
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
        from monet_regrid.utils import format_for_regrid
        from monet_regrid.methods.flox_reduce import statistic_reduce

        ds_formatted = format_for_regrid(self.source_data, self.target_grid, stats=True)

        return statistic_reduce(
            ds_formatted, self.target_grid, time_dim, method, skipna, fill_value
        )

    def most_common(
        self,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
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

        from monet_regrid.utils import format_for_regrid
        from monet_regrid.methods.flox_reduce import compute_mode

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

        from monet_regrid.utils import format_for_regrid
        from monet_regrid.methods.flox_reduce import compute_mode

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
        self,
        source_data: xr.DataArray | xr.Dataset,
        target_grid: xr.Dataset,
        method: str = "linear",
        **kwargs: Any
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
        super().__init__(source_data, target_grid, **kwargs)

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

        # Override with any runtime kwargs
        method = kwargs.get("method", self.method)
        method_kwargs = {**self.method_kwargs, **{k: v for k, v in kwargs.items()
                                                if k not in ["method"]}}

        # Create the CurvilinearInterpolator
        from monet_regrid.curvilinear import CurvilinearInterpolator

        # Create the interpolator with the source and target grids
        interpolator = CurvilinearInterpolator(
            source_grid=self._create_source_grid_from_data(input_data),
            target_grid=self.target_grid,
            method=method,
            **method_kwargs
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
           import cf_xarray
           lat_coord = data.cf["latitude"]
           lon_coord = data.cf["longitude"]
           lat_name = lat_coord.name
           lon_name = lon_coord.name

           # Extract the coordinate variables
           source_grid = xr.Dataset({
               lat_name: data[lat_name],
               lon_name: data[lon_name]
           })

           return source_grid
       except (KeyError, AttributeError):
           # Fallback to manual search
           lat_coords = [name for name in data.coords
                        if "lat" in str(name).lower() or "latitude" in str(name).lower()]
           lon_coords = [name for name in data.coords
                        if "lon" in str(name).lower() or "longitude" in str(name).lower()]

           if lat_coords and lon_coords:
               # If lat/lon coordinates are found in the data, use them
               lat_name = lat_coords[0]
               lon_name = lon_coords[0]

               source_grid = xr.Dataset({
                   lat_name: data[lat_name],
                   lon_name: data[lon_name]
               })

               return source_grid
           else:
               # If no explicit lat/lon coordinates are found in the data,
               # we need to infer the spatial dimensions from the data shape
               # and use the source grid that was provided during initialization
               # In this case, the CurvilinearInterpolator should be initialized differently
               # This is a complex scenario - for now, let's assume that the source grid
               # coordinates were already provided during initialization and we can
               # extract spatial coordinate information from the data dimensions
               # by assuming the last two dimensions are spatial
               if len(data.dims) >= 2:
                   # Use the last two dimensions as spatial dimensions
                   y_dim, x_dim = data.dims[-2], data.dims[-1]

                   # Create simple coordinate arrays based on the spatial dimensions
                   y_coords = np.arange(data.sizes[y_dim])
                   x_coords = np.arange(data.sizes[x_dim])

                   # Create 2D coordinate grids
                   lon_2d, lat_2d = np.meshgrid(x_coords, y_coords)

                   # Create a simple coordinate dataset
                   source_grid = xr.Dataset({
                       "latitude": (["y", "x"], lat_2d),
                       "longitude": (["y", "x"], lon_2d)
                   })

                   return source_grid
               else:
                   raise ValueError("Source data must have at least 2 dimensions for curvilinear regridding")

    def _validate_inputs(self) -> None:
       """Validate the source data and target grid inputs for curvilinear regridding."""
       if not isinstance(self.source_data, (xr.DataArray, xr.Dataset)):
           raise TypeError("source_data must be an xarray DataArray or Dataset")

       if not isinstance(self.target_grid, xr.Dataset):
           raise TypeError("target_grid must be an xarray Dataset")

       # For curvilinear regridders, we only need to validate that the target grid has latitude/longitude coordinates
       # The source data can be passed without explicit lat/lon coordinates, as the grid information
       # is handled separately in the _create_source_grid_from_data method
       try:
           import cf_xarray
           # Use cf-xarray to identify latitude and longitude coordinates in target
           if hasattr(self.target_grid, "cf"):
               target_lat = self.target_grid.cf["latitude"]
               target_lon = self.target_grid.cf["longitude"]
           else:
               # Fallback to manual search
               lat_coords = [name for name in self.target_grid.coords
                            if "lat" in str(name).lower() or "latitude" in str(name).lower()]
               lon_coords = [name for name in self.target_grid.coords
                            if "lon" in str(name).lower() or "longitude" in str(name).lower()]

               if not lat_coords or not lon_coords:
                   raise ValueError("Target grid must have latitude and longitude coordinates")
       except (KeyError, AttributeError):
           # Fallback to manual search
           lat_coords = [name for name in self.target_grid.coords
                        if "lat" in str(name).lower() or "latitude" in str(name).lower()]
           lon_coords = [name for name in self.target_grid.coords
                        if "lon" in str(name).lower() or "longitude" in str(name).lower()]

           if not lat_coords or not lon_coords:
               raise ValueError("Target grid must have latitude and longitude coordinates")

    def to_file(self, filepath: str, **kwargs: Any) -> None:
        """Save the regridder to a file.

        Args:
            filepath: Path to save the regridder
            **kwargs: Additional arguments for file saving
        """
        import pickle

        # Create a serializable representation
        config = {
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "source_data": self.source_data,  # This may need special handling for Dask
            "target_grid": self.target_grid
        }

        with open(filepath, "wb") as f:
            pickle.dump(config, f)

    @classmethod
    def from_file(cls, filepath: str, **kwargs: Any) -> CurvilinearRegridder:
        """Load a regridder from a file.

        Args:
            filepath: Path to load the regridder from
            **kwargs: Additional arguments for file loading

        Returns:
            Instance of CurvilinearRegridder
        """
        import pickle

        with open(filepath, "rb") as f:
            config = pickle.load(f)

        return cls(
            source_data=config["source_data"],
            target_grid=config["target_grid"],
            method=config["method"],
            **config["method_kwargs"]
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
                source_dims = {dim: int(len(self.source_data[dim])) if dim in self.source_data.dims else 0
                              for dim in self.source_data.dims}

        return {
            "type": "CurvilinearRegridder",
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "source_dims": source_dims,
            "target_coords": list(self.target_grid.coords),
            "grid_type": "curvilinear",
            "status": "implemented"
        }
