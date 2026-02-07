"""Implementation of flox reduction based regridding methods."""

from __future__ import annotations

from typing import Any, overload

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr

from monet_regrid import utils
from monet_regrid.methods._shared import (
    construct_intervals,
    reduce_data_to_new_domain,
    restore_properties,
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
URLs updated, and documentation adapted for new branding.
"""


@overload
def statistic_reduce(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
    fill_value: None | Any = None,
) -> xr.DataArray:
    ...


@overload
def statistic_reduce(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
    fill_value: None | Any = None,
) -> xr.Dataset:
    ...


def statistic_reduce(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
    fill_value: None | Any = None,
) -> xr.DataArray | xr.Dataset:
    """Upsampling of data using statistical methods (e.g. the mean or variance).

    We use flox Aggregations to perform a "groupby" over multiple dimensions, which we
    reduce using the specified method.
    https://flox.readthedocs.io/en/latest/aggregations.html

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input data to be regridded. It is assumed that the coordinates are sorted.
    target_ds : xr.Dataset
        Target dataset containing coordinates to regrid to.
    time_dim : str | None
        Name of the time dimension. Use `None` to force regridding over time.
    method : str
        Reduction method (e.g., "sum", "mean", "var", "std", "median", "max", "min").
    skipna : bool, optional
        Whether to ignore NaN values. Defaults to False.
    fill_value : Any, optional
        Value to fill uncovered parts of the target grid. Defaults to None.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The regridded data.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from monet_regrid.methods.flox_reduce import statistic_reduce
    >>> ds = xr.Dataset({"a": (("lat", "lon"), np.random.rand(10, 10))},
    ...                 coords={"lat": np.arange(10), "lon": np.arange(10)})
    >>> target = xr.Dataset(coords={"lat": [2, 5, 8], "lon": [2, 5, 8]})
    >>> res = statistic_reduce(ds, target, time_dim=None, method="mean")
    """
    valid_methods = ["sum", "mean", "var", "std", "median", "max", "min"]
    if method not in valid_methods:
        msg = f"Invalid method. Please choose from '{valid_methods}'."
        raise ValueError(msg)

    # Make sure the regridding coordinates are sorted
    coord_names = utils.common_coords(data, target_ds, remove_coord=time_dim)
    sorted_target_coords = xr.Dataset(coords=target_ds.coords)
    for coord_name in coord_names:
        sorted_target_coords = utils.ensure_monotonic(sorted_target_coords, coord_name)
        data = utils.ensure_monotonic(data, coord_name)
    coords = {name: sorted_target_coords[name] for name in coord_names}

    bounds = tuple(construct_intervals(sorted_target_coords[coord].to_numpy()) for coord in coords)

    data = reduce_data_to_new_domain(data, sorted_target_coords, coord_names)

    result: xr.Dataset = flox.xarray.xarray_reduce(
        data,
        *coords,
        func=method,
        expected_groups=bounds,
        skipna=skipna,
        fill_value=fill_value,
    )

    result = restore_properties(result, data, target_ds, coord_names, fill_value)
    result = result.reindex_like(sorted_target_coords, copy=False)

    # Update history for provenance
    utils.update_history(result, f"Reduced using monet_regrid.methods.flox_reduce.statistic_reduce (method={method})")

    return result


def find_matching_int_dtype(
    a: np.ndarray,
) -> type[np.signedinteger] | type[np.unsignedinteger]:
    """Find the smallest integer datatype that can cover the given array.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    type[np.signedinteger] | type[np.unsignedinteger]
        Smallest compatible integer dtype.
    """
    # Integer types in increasing memory use
    int_types: list[type[np.signedinteger] | type[np.unsignedinteger]] = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
    ]
    for dtype in int_types:
        if (a.max() <= np.iinfo(dtype).max) and (a.min() >= np.iinfo(dtype).min):
            return dtype
    return np.int64


def compute_mode(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    values: np.ndarray,
    time_dim: str | None,
    fill_value: None | Any = None,
    anti_mode: bool = False,
) -> xr.DataArray:
    """Upsample the input data using a "most common label" (mode) approach.

    Parameters
    ----------
    data : xr.DataArray
        Input DataArray with integer dtype.
    target_ds : xr.Dataset
        Target dataset with coordinates to regrid to.
    values : np.ndarray
        Labels expected in the input data.
    time_dim : str | None
        Name of time dimension. Use `None` to force regridding over time.
    fill_value : Any, optional
        Value to fill uncovered parts of the target grid. Defaults to None.
    anti_mode : bool, optional
        If True, find the least-common value (anti-mode). Defaults to False.

    Returns
    -------
    xr.DataArray
        Regridded categorical data.

    Raises
    ------
    ValueError
        If the input data is not of an integer dtype.
    """
    array_name = data.name if data.name is not None else "DATA_NAME"

    # Must be categorical data (integers)
    if not np.issubdtype(data.dtype, np.integer):
        msg = (
            "Your input data has to be of an integer datatype for this method.\n"
            f"    instead, your data is of type '{data.dtype}'."
            "You can convert the data with:\n        `dataset.astype(int)`."
        )
        raise ValueError(msg)

    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    target_coords = xr.Dataset(target_ds.coords)  # stores coords for reindexing later
    sorted_target_coords = target_coords.sortby(coords)

    bounds = tuple(construct_intervals(sorted_target_coords[coord].to_numpy()) for coord in coords)

    data = reduce_data_to_new_domain(data, sorted_target_coords, coords)

    result: xr.DataArray = flox.xarray.xarray_reduce(
        xr.ones_like(data, dtype=bool),
        data,  # important, needs to be int
        *coords,
        dim=coords,
        func="count",
        expected_groups=(pd.Index(values.astype(data)), *bounds),
        fill_value=-1,
    )
    result = result.idxmax(array_name) if not anti_mode else result.idxmin(array_name)

    result = restore_properties(result, data, target_ds, coords, fill_value)
    result = result.reindex_like(target_coords, copy=False)

    # Update history for provenance
    mode_str = "least_common" if anti_mode else "most_common"
    utils.update_history(result, f"Reduced using monet_regrid.methods.flox_reduce.compute_mode ({mode_str})")

    return result
