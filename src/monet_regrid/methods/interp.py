"""
Methods based on xr.interp and efficient scipy interpolators.

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

from collections.abc import Hashable, Sequence
from typing import Literal, overload

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator  # type: ignore


@overload
def interp_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray: ...


@overload
def interp_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.Dataset: ...


def interp_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using xarray's interp method or scipy's RegularGridInterpolator.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        method: Which interpolation method to use (e.g. 'linear', 'nearest').

    Returns:
        Regridded input dataset
    """
    # Identify common coordinates
    coord_names = set(target_ds.coords).intersection(set(data.coords))

    # If the input is a DataArray and we have compatible coordinates, try fast path
    if isinstance(data, xr.DataArray) and len(coord_names) > 0:
        # Check if coordinates are monotonic (required for RegularGridInterpolator)
        # and if we have a dense grid
        try:
            return _interp_regrid_fast(data, target_ds, method, list(coord_names))
        except (ValueError, IndexError, NotImplementedError):
            # Fallback to xarray's interp if fast path fails
            # e.g. if coordinates are not monotonic or other edge cases
            pass

    coords = {name: target_ds[name] for name in coord_names}
    coord_attrs = {coord: data[coord].attrs for coord in coord_names}

    interped = data.interp(
        coords=coords,
        method=method,
    )

    # xarray's interp drops some of the coordinate's attributes (e.g. long_name)
    for coord in coord_names:
        interped[coord].attrs = coord_attrs[coord]

    return interped  # type: ignore


def _interp_regrid_fast(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
    coord_names: Sequence[Hashable],
) -> xr.DataArray:
    """Fast interpolation using scipy.interpolate.RegularGridInterpolator directly.

    This avoids some overhead from xarray's interp() method.
    """
    # Sort coordinate names to match data dimensions order where possible
    # This is critical for RegularGridInterpolator which expects points in (n, D) format

    # Get interpolation dimensions (must be in both data dims and coord_names)
    interp_dims = [dim for dim in data.dims if dim in coord_names]

    if not interp_dims:
        raise ValueError("No interpolation dimensions found")

    # Prepare source coordinates and check monotonicity
    src_coords = []
    for dim in interp_dims:
        coord_vals = data.coords[dim].values
        # RegularGridInterpolator requires strictly increasing coordinates
        # We can handle decreasing by flipping, but mixed is not allowed
        # Check if monotonic increasing
        is_monotonic_inc = np.all(np.diff(coord_vals) > 0)
        # Check if monotonic decreasing
        is_monotonic_dec = np.all(np.diff(coord_vals) < 0)

        if not (is_monotonic_inc or is_monotonic_dec):
            raise ValueError(f"Coordinate {dim} is not monotonic")
        src_coords.append(coord_vals)

    # Prepare target coordinates
    # For RegularGridInterpolator, we need to create a meshgrid of target points
    tgt_coords_1d = []
    for dim in interp_dims:
        tgt_coords_1d.append(target_ds.coords[dim].values)

    # Map method names
    scipy_method = method
    if method == "cubic":
        scipy_method = "cubic"

    # Create interpolator
    # Note: fill_value=np.nan is safer but might be slower; xarray default is usually nan
    interpolator = RegularGridInterpolator(
        tuple(src_coords), data.values, method=scipy_method, bounds_error=False, fill_value=np.nan
    )

    # Generate target points grid
    # We use indexing='ij' to match matrix indexing (row, col, ...)
    tgt_mesh = np.meshgrid(*tgt_coords_1d, indexing="ij")

    # Stack to get shape (N, D) where N is total points and D is dimensions
    # We flatten the meshgrids first
    flat_tgt = np.stack([m.ravel() for m in tgt_mesh], axis=-1)

    # Interpolate
    new_values_flat = interpolator(flat_tgt)

    # Reshape back to target grid shape
    # The shape is determined by the target coordinate lengths
    target_shape = [len(c) for c in tgt_coords_1d]

    # If there are extra dimensions in source data that were not interpolated over
    # (e.g. time), we need to handle them.
    # Current implementation assumes all dims are interpolated or we need to iterate.
    # For now, if data has extra dims, we fall back to xarray interp via the try/except in caller.
    if len(data.dims) != len(interp_dims):
        # We could implement iteration over extra dims here for even more speedup
        # compared to xarray's loop, but for now let's just fallback
        raise NotImplementedError("Extra dimensions not supported in fast path yet")

    new_values = new_values_flat.reshape(target_shape)

    # Create result DataArray
    # We need to construct the new coordinates dict
    new_coords = {}
    for dim in interp_dims:
        new_coords[dim] = target_ds.coords[dim]

    # Add back attributes
    result = xr.DataArray(new_values, dims=interp_dims, coords=new_coords, attrs=data.attrs, name=data.name)

    return result
