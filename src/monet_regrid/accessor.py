from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from monet_regrid.constants import GridType
from monet_regrid.core import BaseRegridder, CurvilinearRegridder, RectilinearRegridder
from monet_regrid.utils import (
    _get_grid_type,
    validate_input,
)

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

    def _prepare_regridder(self, ds_target_grid: xr.Dataset, method: str, time_dim: str | None, **kwargs: Any) -> BaseRegridder:
        """Prepare and build the appropriate regridder."""
        source_grid_type = _get_grid_type(self._obj)
        target_grid_type = _get_grid_type(ds_target_grid)

        if GridType.CURVILINEAR not in (source_grid_type, target_grid_type):
            validated_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        else:
            validated_target_grid = ds_target_grid

        return self.build_regridder(
            ds_target_grid=validated_target_grid,
            method=method,
            source_grid_type=source_grid_type,
            target_grid_type=target_grid_type,
            time_dim=time_dim,
            **kwargs,
        )

    def build_regridder(
        self,
        ds_target_grid: xr.Dataset,
        method: str = "linear",
        source_grid_type: GridType | None = None,
        target_grid_type: GridType | None = None,
        **kwargs: Any,
    ) -> BaseRegridder:
        """Factory method to build the appropriate regridder based on grid type.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        method : str, optional
            The regridding method to use, by default "linear".
        source_grid_type : GridType | None, optional
            The grid type of the source data. If not provided, it will be
            detected automatically.
        target_grid_type : GridType | None, optional
            The grid type of the target data. If not provided, it will be
            detected automatically.
        **kwargs : Any
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        BaseRegridder
            An instance of the appropriate regridder class based on grid type.
        """
        if source_grid_type is None:
            source_grid_type = _get_grid_type(self._obj)
        if target_grid_type is None:
            target_grid_type = _get_grid_type(ds_target_grid)

        if GridType.CURVILINEAR in (source_grid_type, target_grid_type):
            return CurvilinearRegridder(source_data=self._obj, target_grid=ds_target_grid, method=method, **kwargs)
        return RectilinearRegridder(source_data=self._obj, target_grid=ds_target_grid, method=method, **kwargs)

    def linear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with linear interpolation.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        **kwargs : Any
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data regridded to the target dataset coordinates.
        """
        regridder = self._prepare_regridder(ds_target_grid, "linear", time_dim, **kwargs)
        return regridder()

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target with nearest-neighbor interpolation.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        **kwargs : Any
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data regridded to the target dataset coordinates.
        """
        regridder = self._prepare_regridder(ds_target_grid, "nearest", time_dim, **kwargs)
        return regridder()

    def bilinear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with bilinear interpolation.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        **kwargs : Any
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data regridded to the target dataset coordinates.
        """
        regridder = self._prepare_regridder(ds_target_grid, "bilinear", time_dim, **kwargs)
        return regridder()

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with cubic interpolation.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        **kwargs : Any
            Additional keyword arguments to pass to the regridder.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data regridded to the target dataset coordinates.
        """
        regridder = self._prepare_regridder(ds_target_grid, "cubic", time_dim, **kwargs)
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

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Dataset containing the target coordinates.
        latitude_coord : str | None, optional
            Name of the latitude coord, to be used for applying the
            spherical correction. By default, attempt to infer a latitude coordinate
            as either "latitude" or "lat".
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        skipna : bool, optional
            If True, enable handling for NaN values. This adds only a small
            amount of overhead, but can be disabled for optimal performance on data
            without any NaNs. Defaults to True.
        nan_threshold : float, optional
            Threshold value that will retain any output points
            containing at least this many non-null input points. The default value
            is 1.0, which will keep output points containing any non-null inputs,
            while a value of 0.0 will only keep output points where all inputs are
            non-null.
        output_chunks : dict[Hashable, int] | None, optional
            Optional dictionary of explicit chunk sizes for the output
            data. If not provided, the output will be chunked the same as the input
            data.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data regridded to the target dataset coordinates.
        """
        if not 0.0 <= nan_threshold <= 1.0:
            msg = "nan_threshold must be between [0, 1]]"
            raise ValueError(msg)

        regridder = self._prepare_regridder(
            ds_target_grid,
            "conservative",
            time_dim,
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

        Note that in the case of two unique values with the same count, the behaviour
        is not deterministic, and the resulting "most common" one will randomly be
        either of the two.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Target grid dataset.
        values : np.ndarray
            Numpy array containing all labels expected to be in the
            input data. For example, `np.array([0, 2, 4])`, if the data only
            contains the values 0, 2 and 4.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        fill_value : Any | None, optional
            What value to fill uncovered parts of the target grid.
            By default this will be NaN, and integer type data will be cast to
            float to accommodate this.
        nan_threshold : float, optional
            Threshold for the nan_threshold, by default 1.0.

        Returns
        -------
        xr.DataArray
            Regridded data.
        """
        regridder = self._prepare_regridder(ds_target_grid, "most_common", time_dim)
        if hasattr(regridder, "most_common"):
            return regridder.most_common(values, time_dim, fill_value, nan_threshold=nan_threshold)
        msg = f"{regridder.__class__.__name__} does not support 'most_common' method."
        raise AttributeError(msg)

    def least_common(
        self,
        ds_target_grid: xr.Dataset,
        values: np.ndarray,
        time_dim: str | None = "time",
        fill_value: None | Any = None,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray:
        """Regrid by taking the least common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unique values with the same count, the behaviour
        is not deterministic, and the resulting "least common" one will randomly be
        either of the two.

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Target grid dataset.
        values : np.ndarray
            Numpy array containing all labels expected to be in the
            input data. For example, `np.array([0, 2, 4])`, if the data only
            contains the values 0, 2 and 4.
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        fill_value : Any | None, optional
            What value to fill uncovered parts of the target grid.
            By default this will be NaN, and integer type data will be cast to
            float to accommodate this.
        nan_threshold : float, optional
            Threshold for the nan_threshold, by default 1.0.

        Returns
        -------
        xr.DataArray
            Regridded data.
        """
        regridder = self._prepare_regridder(ds_target_grid, "least_common", time_dim)
        if hasattr(regridder, "least_common"):
            return regridder.least_common(values, time_dim, fill_value, nan_threshold=nan_threshold)
        msg = f"{regridder.__class__.__name__} does not support 'least_common' method."
        raise AttributeError(msg)

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

        Parameters
        ----------
        ds_target_grid : xr.Dataset
            Target grid dataset.
        method : str
            One of the following reduction methods: "sum", "mean", "var", "std",
            "median", "min", or "max".
        time_dim : str | None, optional
            Name of the time dimension, by default "time". Use `None` to
            force regridding over the time dimension.
        skipna : bool, optional
            If NaN values should be ignored, by default False.
        fill_value : Any | None, optional
            What value to fill uncovered parts of the target grid.
            By default this will be NaN, and integer type data will be cast to
            float to accommodate this.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Object with regridded land cover categorical data.
        """
        regridder = self._prepare_regridder(ds_target_grid, "stat", time_dim)
        if hasattr(regridder, "stat"):
            return regridder.stat(method, time_dim, skipna, fill_value)
        msg = f"{regridder.__class__.__name__} does not support 'stat' method."
        raise AttributeError(msg)
