"""
Utility functions shared between methods.

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

import warnings
from collections.abc import Hashable
from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr

from monet_regrid import utils


def construct_intervals(coord: np.ndarray) -> pd.IntervalIndex:
    """Create pandas.intervals with given coordinates.

    Parameters
    ----------
    coord : np.ndarray
        Array of coordinate centers.

    Returns
    -------
    pd.IntervalIndex
        Intervals representing the bins.
    """
    step_size = np.median(np.diff(coord, n=1))
    breaks = np.append(coord, coord[-1] + step_size) - step_size / 2

    # Note: closed="both" triggers an `NotImplementedError`
    return pd.IntervalIndex.from_breaks(breaks, closed="left")


@overload
def restore_properties(
    result: xr.DataArray,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[Hashable],
    fill_value: Any,
) -> xr.DataArray:
    ...


@overload
def restore_properties(
    result: xr.Dataset,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[Hashable],
    fill_value: Any,
) -> xr.Dataset:
    ...


def restore_properties(
    result: xr.DataArray | xr.Dataset,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[Hashable],
    fill_value: Any,
) -> xr.DataArray | xr.Dataset:
    """Restore coord names, copy values and attributes of target, & add NaN padding.

    Parameters
    ----------
    result : xr.DataArray | xr.Dataset
        The raw reduced data from flox.
    original_data : xr.DataArray | xr.Dataset
        The original input data before reduction.
    target_ds : xr.Dataset
        The target grid dataset.
    coords : list[Hashable]
        List of coordinate names that were reduced.
    fill_value : Any
        Value used to fill uncovered regions.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The processed data with restored metadata and alignment.
    """
    result.attrs = original_data.attrs

    # Map target coordinates to source coordinates for coverage check
    try:
        src_lat, src_lon = utils.identify_cf_coordinates(original_data)
        tgt_lat, tgt_lon = utils.identify_cf_coordinates(target_ds)
        src_map = {tgt_lat: src_lat, tgt_lon: src_lon}
    except ValueError:
        src_map = {}

    result = result.rename({f"{coord}_bins": coord for coord in coords})
    for coord in coords:
        result[coord] = target_ds[coord]
        result[coord].attrs = target_ds[coord].attrs

        # Replace zeros outside of original data grid with NaNs
        src_coord = src_map.get(coord, coord)
        if src_coord in original_data.coords:
            covered = (target_ds[coord] <= original_data[src_coord].max()) & (target_ds[coord] >= original_data[src_coord].min())
        else:
            covered = xr.DataArray(True)

        if (~covered).any():
            if fill_value is None:
                if np.issubdtype(result.dtype, np.integer):
                    msg = (
                        "No fill_value is provided; data will be cast to "
                        "floating point dtype to be able to use NaN for missing values."
                    )
                    warnings.warn(msg, stacklevel=1)
                result = result.where(covered)
            else:
                result = result.where(covered, fill_value)

    # Determine the desired dimension order. We try to match the original
    # dimension order, but need to account for dimensions that were renamed
    # during regridding (e.g., in the curvilinear case).
    original_spatial_dims = [d for d in original_data.dims if d not in result.dims]
    new_spatial_dims = [d for d in result.dims if d not in original_data.dims]

    target_dims = list(original_data.dims)
    if original_spatial_dims:
        # Find index of first original spatial dim
        idx = target_dims.index(original_spatial_dims[0])
        # Remove all original spatial dims
        for d in original_spatial_dims:
            if d in target_dims:
                target_dims.remove(d)
        # Insert new spatial dims at that index
        for i, d in enumerate(new_spatial_dims):
            target_dims.insert(idx + i, d)
    else:
        # If no original dims were removed, just use result.dims but try to match order
        target_dims = [d for d in original_data.dims if d in result.dims]
        for d in result.dims:
            if d not in target_dims:
                target_dims.append(d)

    # Filter to only existing dims in result
    target_dims = [d for d in target_dims if d in result.dims]

    return result.transpose(*target_dims)


@overload
def reduce_data_to_new_domain(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    coords: list[Hashable],
) -> xr.DataArray:
    ...


@overload
def reduce_data_to_new_domain(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[Hashable],
) -> xr.Dataset:
    ...


def reduce_data_to_new_domain(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[Hashable],
) -> xr.DataArray | xr.Dataset:
    """Slice the input data to bounds of the target dataset, to reduce computations.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input data to be sliced.
    target_ds : xr.Dataset
        Target dataset providing the spatial bounds.
    coords : list[Hashable]
        Names of coordinates to slice along.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The sliced data.
    """
    for coord in coords:
        coord_diff = target_ds[coord].diff(coord)
        coord_res = coord_diff.median().values.item()
        c_min = target_ds[coord].min().values.item()
        c_max = target_ds[coord].max().values.item()
        data = data.sel(
            {
                coord: slice(
                    float(c_min) - coord_res,
                    float(c_max) + coord_res,
                )
            }
        )
    return data
