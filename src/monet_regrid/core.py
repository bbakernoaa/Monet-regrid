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
from typing import Any

import cf_xarray  # noqa: F401
import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator
from monet_regrid.methods import conservative, interp
from monet_regrid.methods.flox_reduce import compute_mode, statistic_reduce
from monet_regrid.utils import (
    _create_cache_key,
    _create_lat_lon_from_dims,
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
        """Initialize the regridder.

        Parameters
        ----------
        source_data : xr.DataArray | xr.Dataset | None
            The source data to be regridded. If None, a data-agnostic regridder
            is created which can be applied to data later.
        target_grid : xr.Dataset
            The target grid specification.
        """
        self.source_data = source_data
        self.target_grid = target_grid
        self._validate_inputs()

    @abc.abstractmethod
    def __call__(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for the regridding operation.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The regridded data.
        """
        pass

    @abc.abstractmethod
    def _get_config(self) -> dict[str, Any]:
        """Get the configuration of the regridder for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the regridder's configuration,
            including module and class names for dynamic instantiation.
        """
        pass

    def to_file(self, filepath: str) -> None:
        """Save the regridder configuration to a NetCDF file.

        This method saves the regridder's configuration and the target grid to a
        NetCDF file. The source data is intentionally not saved, allowing the
        regridder to be reused with different source datasets.

        Parameters
        ----------
        filepath : str
            Path to save the regridder configuration.
        """
        config = self._get_config()
        self.target_grid.attrs["regridder_config"] = json.dumps(config)
        self.target_grid.to_netcdf(filepath)

    @classmethod
    def from_file(cls, filepath: str) -> BaseRegridder:
        """Load a regridder from a NetCDF file.
        This class method reconstructs a regridder from a NetCDF file that was
        created with the `to_file` method. It loads the target grid and the
        regridding configuration, creating a "data-agnostic" regridder that
        can be applied to new xarray DataArrays or Datasets.

        Parameters
        ----------
        filepath : str
            Path to the NetCDF file.

        Returns
        -------
        BaseRegridder
            An instance of a regridder class, initialized with `source_data=None`.
        """
        with xr.open_dataset(filepath) as ds:
            target_grid = ds.copy(deep=True)

        config_str = target_grid.attrs.pop("regridder_config", None)
        if not config_str:
            msg = "regridder_config attribute not found in the file."
            raise ValueError(msg)
        config = json.loads(config_str)

        # Handle new format (dynamic import) and old format (hardcoded) for backward compatibility
        if "module" in config and "class" in config:
            module_name = config.pop("module")
            class_name = config.pop("class")

            try:
                module = __import__(module_name, fromlist=[class_name])
                regridder_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                msg = f"Failed to load regridder class '{class_name}' from module '{module_name}'."
                raise ImportError(msg) from e
        elif "regridder_type" in config:
            regridder_type = config.pop("regridder_type")
            if regridder_type == "RectilinearRegridder":
                regridder_class = RectilinearRegridder
            elif regridder_type == "CurvilinearRegridder":
                regridder_class = CurvilinearRegridder
            else:
                msg = f"Unknown regridder type: {regridder_type}"
                raise ValueError(msg)
        else:
            msg = (
                "Could not determine regridder type from file. Missing 'regridder_type' "
                "or 'module'/'class' from config."
            )
            raise ValueError(msg)

        if not issubclass(regridder_class, cls):
            msg = f"Loaded class {regridder_class} is not a subclass of {cls}."
            raise TypeError(msg)

        return regridder_class(source_data=None, target_grid=target_grid, **config)

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Get information about the regridder instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing regridder metadata and configuration.
        """
        pass

    def _validate_inputs(self) -> None:
        """Validate the source data and target grid inputs."""
        # When source_data is None, we skip validation related to it.
        # This allows for the creation of a data-agnostic regridder.
        if self.source_data is not None:
            if not isinstance(self.source_data, (xr.DataArray, xr.Dataset)):
                msg = "source_data must be an xarray DataArray or Dataset"
                raise TypeError(msg)

        if not isinstance(self.target_grid, xr.Dataset):
            msg = "target_grid must be an xarray Dataset"
            raise TypeError(msg)

        # Use a centralized coordinate identification function
        if self.source_data is not None:
            try:
                self.source_lat_name, self.source_lon_name = identify_cf_coordinates(self.source_data)
            except ValueError as e:
                msg = f"Source data validation failed: {e}"
                raise ValueError(msg) from e

        try:
            self.target_lat_name, self.target_lon_name = identify_cf_coordinates(self.target_grid)
        except ValueError as e:
            msg = f"Target grid validation failed: {e}"
            raise ValueError(msg) from e

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the regridder for serialization (Dask compatibility)."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the regridder from serialized state (Dask compatibility)."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        """Return a string representation of the regridder."""
        return f"<{self.__class__.__name__}>\n{self.info()}"


class RectilinearRegridder(BaseRegridder):
    """Regridder implementation for rectilinear grids using interpolation methods.

    This class handles regridding between rectilinear grids using various interpolation
    methods like linear, nearest-neighbor, bilinear, cubic, and conservative approaches.
    """

    def __init__(
        self,
        source_data: xr.DataArray | xr.Dataset | None,
        target_grid: xr.Dataset,
        method: str = "linear",
        time_dim: str | None = "time",
        **kwargs: Any,
    ):
        """Initialize the rectilinear regridder.

        Parameters
        ----------
        source_data : xr.DataArray | xr.Dataset | None
            The source data to be regridded.
        target_grid : xr.Dataset
            The target grid specification.
        method : str, optional
            Interpolation method. Defaults to "linear".
        time_dim : str | None, optional
            Name of the time dimension. Defaults to "time".
        **kwargs : Any
            Additional method-specific arguments.
        """
        self.method = method
        self.time_dim = time_dim
        self.method_kwargs = kwargs
        # Add caching for validated target grid and formatted data
        self._validation_cache: dict[tuple, xr.Dataset] = {}
        self._formatting_cache: dict[tuple, xr.DataArray | xr.Dataset] = {}
        super().__init__(source_data, target_grid)

    def __call__(self, data: xr.DataArray | xr.Dataset | None = None, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation.

        This method performs the regridding of the source data to the target grid
        using the specified interpolation method. It can be called with new data
        or will use the data provided during initialization.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset | None, optional
            The data to be regridded. If None, the `source_data` provided
            during initialization is used. Defaults to None.
        **kwargs : Any
            Additional keyword arguments to override the regridder's
            initialization parameters for this specific call. For example,
            `method='nearest'` could be used to temporarily change the
            interpolation method.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The regridded data, with the same type as the input `data`.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> source_da = xr.DataArray(
        ...     np.random.rand(10, 20),
        ...     dims=["y", "x"],
        ...     coords={"lat": (("y",), np.arange(0, 10)), "lon": (("x",), np.arange(0, 20))},
        ... )
        >>> target_ds = xr.Dataset(
        ...     coords={
        ...         "lat": (("y_new",), np.arange(0.5, 10, 2)),
        ...         "lon": (("x_new",), np.arange(0.5, 20, 2)),
        ...     }
        ... )
        >>> regridder = RectilinearRegridder(source_da, target_ds, method="linear")
        >>> regridded_da = regridder()
        >>> print(regridded_da.shape)
        (5, 10)
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
            regridded_data = interp.interp_regrid(formatted_data, validated_target_grid, method)
        elif method == "conservative":
            # Handle conservative regridding with its specific parameters
            latitude_coord = method_kwargs.get("latitude_coord", None)
            skipna = method_kwargs.get("skipna", True)
            nan_threshold = method_kwargs.get("nan_threshold", 1.0)
            output_chunks = method_kwargs.get("output_chunks", None)

            regridded_data = conservative.conservative_regrid(
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

        # Update history attribute for provenance
        history_message = f"Regridded using RectilinearRegridder with method='{method}'"
        existing_history = regridded_data.attrs.get("history", "")
        regridded_data.attrs["history"] = (
            f"{existing_history}\n{history_message}" if existing_history else history_message
        )

        return regridded_data

    def _get_config(self) -> dict[str, Any]:
        """Get the configuration of the regridder for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the regridder's configuration.
        """
        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "method": self.method,
            "time_dim": self.time_dim,
            "method_kwargs": self.method_kwargs,
        }

    def info(self) -> dict[str, Any]:
        """Get information about the rectilinear regridder instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing regridder metadata and configuration.
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
        """Upsample data using statistical methods.

        Parameters
        ----------
        method : str
            The reduction method, e.g., "sum", "mean", "min", "max".
        time_dim : str | None, optional
            Name of the time dimension. Defaults to "time".
        skipna : bool, optional
            If True, ignores NaN values. Defaults to False.
        fill_value : Any, optional
            Fill value for uncovered target grid parts. Defaults to None.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The regridded data.
        """

        ds_formatted = format_for_regrid(self.source_data, self.target_grid, stats=True)

        return statistic_reduce(ds_formatted, self.target_grid, time_dim, method, skipna, fill_value)

    def most_common(
        self,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
        nan_threshold: float = 1.0,  # noqa: ARG002
    ) -> xr.DataArray:
        """Regrid by taking the most common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "most common" one will randomly be
        either of the two.

        Parameters
        ----------
        values : np.ndarray
            Numpy array containing all labels expected in the input data.
        time_dim : str | None, optional
            Name of the time dimension. Defaults to "time".
        fill_value : Any, optional
            Fill value for uncovered target grid parts. Defaults to None.
        nan_threshold : float, optional
            Threshold for NaN values. Defaults to 1.0.

        Returns
        -------
        xr.DataArray
            The regridded data.
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
        nan_threshold: float = 1.0,  # noqa: ARG002
    ) -> xr.DataArray:
        """Regrid by taking the least common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "least common" one will randomly be
        either of the two.

        Parameters
        ----------
        values : np.ndarray
            Numpy array containing all labels expected in the input data.
        time_dim : str | None, optional
            Name of the time dimension. Defaults to "time".
        fill_value : Any, optional
            Fill value for uncovered target grid parts. Defaults to None.
        nan_threshold : float, optional
            Threshold for NaN values. Defaults to 1.0.

        Returns
        -------
        xr.DataArray
            The regridded data.
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
        self,
        source_data: xr.DataArray | xr.Dataset | None,
        target_grid: xr.Dataset,
        method: str = "linear",
        **kwargs: Any,
    ):
        """Initialize the curvilinear regridder.

        Parameters
        ----------
        source_data : xr.DataArray | xr.Dataset | None
            The source data to be regridded.
        target_grid : xr.Dataset
            The target grid specification.
        method : str, optional
            Interpolation method. Defaults to "linear".
        **kwargs : Any
            Additional method-specific arguments.
        """
        self.method = method
        self.method_kwargs = kwargs
        # Unpack nested method_kwargs that can occur during deserialization
        if "method_kwargs" in self.method_kwargs and len(self.method_kwargs) == 1:
            self.method_kwargs = self.method_kwargs["method_kwargs"]
        self._interpolator_cache: dict[tuple, CurvilinearInterpolator] = {}
        super().__init__(source_data, target_grid)

    def _validate_inputs(self) -> None:
        """Validate inputs, attempting to identify source coordinates gracefully.

        This method validates input data types and attempts to identify source
        coordinates if they exist. If source coordinates are not found, it fails
        silently, allowing for lazy coordinate generation in a later step.

        Raises
        ------
        TypeError
            If `source_data` is not a valid xarray object or `target_grid`
            is not an xarray Dataset.
        ValueError
            If the target grid's coordinates cannot be identified.
        """
        if self.source_data is not None and not isinstance(
            self.source_data, (xr.DataArray, xr.Dataset)
        ):
            msg = "source_data must be an xarray DataArray or Dataset"
            raise TypeError(msg)

        if not isinstance(self.target_grid, xr.Dataset):
            msg = "target_grid must be an xarray Dataset"
            raise TypeError(msg)

        if self.source_data is not None:
            try:
                self.source_lat_name, self.source_lon_name = identify_cf_coordinates(
                    self.source_data
                )
            except ValueError:
                pass

        try:
            self.target_lat_name, self.target_lon_name = identify_cf_coordinates(
                self.target_grid
            )
        except ValueError as e:
            msg = f"Target grid validation failed: {e}"
            raise ValueError(msg) from e

    def __call__(self, data: xr.DataArray | xr.Dataset | None = None, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        """Execute the regridding operation for curvilinear grids.

        This method caches the underlying ``CurvilinearInterpolator`` object after its
        first creation for a given grid and method configuration. Subsequent calls with
        the same configuration will reuse the cached interpolator to improve performance.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset | None, optional
            Data to regrid. Defaults to the source data from initialization.
        **kwargs : Any
            Additional arguments to override initialization parameters.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The regridded data.
        """
        # Use provided data or fall back to source data
        input_data = data if data is not None else self.source_data

        # Override with any runtime kwargs
        method = kwargs.get("method", self.method)
        method_kwargs = {**self.method_kwargs, **{k: v for k, v in kwargs.items() if k not in ["method"]}}

        # Create a stable cache key from the grids and method config
        cache_key = (
            _create_cache_key(input_data),
            _create_cache_key(self.target_grid),
            method,
            tuple(sorted(method_kwargs.items())),
        )

        # Check if we have a cached interpolator
        if cache_key in self._interpolator_cache:
            interpolator = self._interpolator_cache[cache_key]
        else:
            # Identify source coordinates just-in-time from the input data
            source_lat_name, source_lon_name = identify_cf_coordinates(input_data)

            # Create the source grid and the interpolator
            source_grid = self._create_source_grid_from_data(input_data)
            interpolator = CurvilinearInterpolator(
                source_grid=source_grid,
                target_grid=self.target_grid,
                source_lat_name=source_lat_name,
                source_lon_name=source_lon_name,
                target_lat_name=self.target_lat_name,
                target_lon_name=self.target_lon_name,
                method=method,
                **method_kwargs,
            )
            # Cache the interpolator
            self._interpolator_cache[cache_key] = interpolator

        # Apply the interpolation to the actual data
        result = interpolator(input_data)

        # Update history attribute for provenance
        history_message = f"Regridded using CurvilinearRegridder with method='{method}'"
        existing_history = result.attrs.get("history", "")
        result.attrs["history"] = f"{existing_history}\n{history_message}" if existing_history else history_message

        return result

    def _create_source_grid_from_data(
        self, source_data: xr.DataArray | xr.Dataset | None = None
    ) -> xr.Dataset:
        """Create a grid specification from source data, with lazy-loading support.

        This method extracts or generates coordinate information from the source
        data. It first attempts to find explicit CF-compliant latitude/longitude
        coordinates. If none are found, it falls back to generating a lazy
        coordinate grid based on the spatial dimensions of the data using the
        `_create_lat_lon_from_dims` utility.

        Parameters
        ----------
        source_data : xr.DataArray | xr.Dataset | None, optional
            The source data to process. If None, the data from initialization
            is used. Defaults to None.

        Returns
        -------
        xr.Dataset
            A dataset containing the latitude and longitude coordinates.
        """
        data = source_data if source_data is not None else self.source_data

        try:
            lat_name, lon_name = identify_cf_coordinates(data)
            return xr.Dataset({lat_name: data[lat_name], lon_name: data[lon_name]})
        except ValueError:
            return _create_lat_lon_from_dims(data)

    def _get_config(self) -> dict[str, Any]:
        """Get the configuration of the regridder for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the regridder's configuration.
        """
        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "method": self.method,
            "method_kwargs": self.method_kwargs,
        }

    def info(self) -> dict[str, Any]:
        """Get information about the curvilinear regridder instance.

        Returns
        -------
        dict[str, Any]
            Dictionary containing regridder metadata and configuration.
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
