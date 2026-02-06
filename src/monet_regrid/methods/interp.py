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

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal, overload

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


@overload
def interp_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic", "bilinear"],
) -> xr.DataArray:
    ...


@overload
def interp_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic", "bilinear"],
) -> xr.Dataset:
    ...


def interp_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic", "bilinear"],
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using xarray's interp method or scipy's RegularGridInterpolator.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input data to be regridded.
    target_ds : xr.Dataset
        Target dataset containing the coordinates to regrid to.
    method : Literal["linear", "nearest", "cubic", "bilinear"]
        Interpolation method to use.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The regridded data.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from monet_regrid.methods.interp import interp_regrid
    >>> ds = xr.Dataset({"a": (("x", "y"), np.random.rand(10, 10))},
    ...                 coords={"x": np.arange(10), "y": np.arange(10)})
    >>> target = xr.Dataset(coords={"x": np.linspace(0, 9, 20), "y": np.linspace(0, 9, 20)})
    >>> res = interp_regrid(ds, target, method="linear")
    """
    # Identify common coordinates
    coord_names = set(target_ds.coords).intersection(set(data.coords))

    # Handle dimensions present in the target but not the source
    missing_dims = set(target_ds.dims) - set(data.dims)
    if missing_dims:
        for dim in missing_dims:
            if dim in target_ds.coords:
                data = data.expand_dims({dim: target_ds[dim]})

    # Attempt fast path for DataArray or Dataset
    if len(coord_names) > 0:
        try:
            if isinstance(data, xr.DataArray):
                interped = _interp_regrid_fast(data, target_ds, method, list(coord_names))
            else:
                # For Dataset, try fast path for each data variable and coordinate (Scientific Hygiene)
                # We interpolate everything that depends on the interpolated dimensions
                new_vars = {}
                # Handle all variables (data_vars and coords)
                for var_name in list(data.data_vars) + [c for c in data.coords if c not in data.dims]:
                    da = data[var_name]
                    if any(dim in da.dims for dim in coord_names):
                        try:
                            new_vars[var_name] = _interp_regrid_fast(da, target_ds, method, list(coord_names))
                        except (ValueError, IndexError, NotImplementedError):
                            # Fallback for this specific variable
                            interp_dict = {dim: target_ds[dim] for dim in da.dims if dim in coord_names}
                            new_vars[var_name] = da.interp(interp_dict, method=method)
                    else:
                        new_vars[var_name] = da

                # Create the new dataset
                interped = xr.Dataset(new_vars, attrs=data.attrs)
                # Ensure dimension coordinates from target_ds are correctly assigned
                interped = interped.assign_coords({c: target_ds[c] for c in coord_names})

            # Update history for provenance (Fast Path)
            _update_history(interped, method)
            return interped
        except (ValueError, IndexError, NotImplementedError):
            # Fallback to xarray's interp if fast path fails globally
            pass

    # Map coordinate names to dimension names for the interpolation
    coords = {data[name].dims[0]: target_ds[name] for name in coord_names if name in data.coords}
    coord_attrs = {coord: data[coord].attrs for coord in coord_names if coord in data.coords}

    # Perform the interpolation using dimension names
    interped = data.interp(
        coords=coords,
        method=method,
    )

    # xarray's interp drops some of the coordinate's attributes (e.g. long_name)
    for coord in coord_names:
        if coord in interped.coords:
            interped[coord].attrs = coord_attrs[coord]

    # Update history for provenance (Slow Path)
    _update_history(interped, method)

    return interped


def _update_history(obj: xr.DataArray | xr.Dataset, method: str) -> None:
    """Update the history attribute of an xarray object for provenance tracking.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The xarray object to update.
    method : str
        The interpolation method used (e.g., 'linear', 'nearest').

    Returns
    -------
    None
        The object is modified in-place.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 2, 3], attrs={"history": "Original"})
    >>> _update_history(da, "linear")
    >>> print(da.attrs["history"])
    Original
    Interpolated using monet_regrid.methods.interp.interp_regrid (method=linear)
    """
    history = f"Interpolated using monet_regrid.methods.interp.interp_regrid (method={method})"
    existing_history = obj.attrs.get("history", "")
    obj.attrs["history"] = f"{existing_history}\n{history}" if existing_history else history


def _interp_regrid_fast(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic", "bilinear"],
    coord_names: Sequence[Hashable],
) -> xr.DataArray:
    """Fast interpolation using scipy.interpolate.RegularGridInterpolator directly.

    This avoids some overhead from xarray's interp() method by working directly
    on NumPy arrays.

    Parameters
    ----------
    data : xr.DataArray
        Input DataArray to be regridded.
    target_ds : xr.Dataset
        Target dataset containing coordinates.
    method : Literal["linear", "nearest", "cubic", "bilinear"]
        Interpolation method.
    coord_names : Sequence[Hashable]
        Names of coordinates to interpolate over.

    Returns
    -------
    xr.DataArray
        The interpolated DataArray.

    Raises
    ------
    ValueError
        If interpolation dimensions are not found or coordinates are not monotonic.
    NotImplementedError
        If data is Dask-backed or has extra dimensions.
    """
    # Sort coordinate names to match data dimensions order where possible
    # This is critical for RegularGridInterpolator which expects points in (n, D) format

    # Get interpolation dimensions (must be in both data dims and coord_names)
    interp_dims = [dim for dim in data.dims if dim in coord_names]

    if not interp_dims:
        msg = "No interpolation dimensions found"
        raise ValueError(msg)

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
            msg = f"Coordinate {dim} is not monotonic"
            raise ValueError(msg)
        src_coords.append(coord_vals)

    # Prepare target coordinates
    # For RegularGridInterpolator, we need to create a meshgrid of target points
    tgt_coords_1d = []
    for dim in interp_dims:
        tgt_coords_1d.append(target_ds.coords[dim].values)

    # Map method names
    scipy_method = method
    if method == "bilinear":
        scipy_method = "linear"  # scipy uses 'linear' for bilinear in 2D

    # We must be careful not to trigger eager loading if data is dask-backed
    if data.chunks is not None:
        msg = "Dask-backed data not supported in fast path; falling back to xarray.interp"
        raise NotImplementedError(msg)

    # Handle extra dimensions (e.g., time, level) using vectorization
    extra_dims = [d for d in data.dims if d not in interp_dims]
    if extra_dims:
        # Transpose to put interpolation dims first
        transpose_order = interp_dims + extra_dims
        data_transposed = data.transpose(*transpose_order)
        values_to_interp = data_transposed.values
    else:
        values_to_interp = data.values

    # Create interpolator
    # Note: fill_value=np.nan is safer but might be slower; xarray default is usually nan
    interpolator = RegularGridInterpolator(
        tuple(src_coords), values_to_interp, method=scipy_method, bounds_error=False, fill_value=np.nan
    )

    # Generate target points grid
    # We use indexing='ij' to match matrix indexing (row, col, ...)
    tgt_mesh = np.meshgrid(*tgt_coords_1d, indexing="ij")

    # Stack to get shape (N, D) where N is total points and D is dimensions
    flat_tgt = np.stack([m.ravel() for m in tgt_mesh], axis=-1)

    # Interpolate
    new_values_flat = interpolator(flat_tgt)

    # Reshape back to target grid shape
    # The shape is determined by the target coordinate lengths plus extra dims
    target_shape = [len(c) for c in tgt_coords_1d]
    if extra_dims:
        target_shape.extend([data.sizes[d] for d in extra_dims])

    new_values = new_values_flat.reshape(target_shape)

    # Construct the result DataArray with ALL coordinates (Scientific Hygiene)
    new_coords = {}
    result_dims = list(interp_dims) + extra_dims

    for name, coord in data.coords.items():
        if name in interp_dims:
            # Dimension coordinate being interpolated
            new_coords[name] = target_ds.coords[name]
        elif any(dim in coord.dims for dim in interp_dims):
            # Auxiliary coordinate depending on interpolated dims -> Interpolate it
            interp_dict = {dim: target_ds.coords[dim] for dim in coord.dims if dim in interp_dims}
            new_coords[name] = coord.interp(interp_dict, method=method)
        else:
            # Preserve coordinates that don't depend on interpolated dims
            new_coords[name] = coord

    result = xr.DataArray(new_values, dims=result_dims, coords=new_coords, attrs=data.attrs, name=data.name)

    # Transpose back to original dimension order if necessary
    if extra_dims:
        result = result.transpose(*data.dims)

    return result
