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

from monet_regrid import utils


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
            utils.update_history(interped, f"Interpolated using monet_regrid.methods.interp.interp_regrid (method={method})")
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
    utils.update_history(interped, f"Interpolated using monet_regrid.methods.interp.interp_regrid (method={method})")

    return interped


def _interp_regrid_fast(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic", "bilinear"],
    coord_names: Sequence[Hashable],
) -> xr.DataArray:
    """Fast interpolation using scipy.interpolate.RegularGridInterpolator directly.

    This avoids some overhead from xarray's interp() method by working directly
    on NumPy arrays or Dask chunks using xr.apply_ufunc.

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
    """
    # Get interpolation dimensions (must be in both data dims and coord_names)
    interp_dims = [dim for dim in data.dims if dim in coord_names]

    if not interp_dims:
        msg = "No interpolation dimensions found"
        raise ValueError(msg)

    # Prepare source coordinates and check monotonicity
    src_coords = []
    for dim in interp_dims:
        coord_vals = data.coords[dim].values
        is_monotonic_inc = np.all(np.diff(coord_vals) > 0)
        is_monotonic_dec = np.all(np.diff(coord_vals) < 0)

        if not (is_monotonic_inc or is_monotonic_dec):
            msg = f"Coordinate {dim} is not monotonic"
            raise ValueError(msg)
        src_coords.append(coord_vals)

    # Prepare target coordinates
    tgt_coords_1d = [target_ds.coords[dim].values for dim in interp_dims]
    target_shape = tuple(len(c) for c in tgt_coords_1d)

    # Map method names
    scipy_method = method
    if method == "bilinear":
        scipy_method = "linear"

    # Define the output core dimensions and sizes for apply_ufunc
    output_core_dims = [interp_dims]
    output_sizes = {dim: len(target_ds.coords[dim]) for dim in interp_dims}

    # Use xr.apply_ufunc to handle both NumPy and Dask arrays
    result = xr.apply_ufunc(
        _scipy_interp_wrapper,
        data,
        kwargs={
            "src_coords": tuple(src_coords),
            "tgt_coords_1d": tuple(tgt_coords_1d),
            "method": scipy_method,
            "target_shape": target_shape,
        },
        input_core_dims=[interp_dims],
        output_core_dims=output_core_dims,
        exclude_dims=set(interp_dims),
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": output_sizes},
        keep_attrs=True,
    )

    # Construct the result DataArray with ALL coordinates (Scientific Hygiene)
    new_coords = {}
    for name, coord in data.coords.items():
        if name in interp_dims:
            new_coords[name] = target_ds.coords[name]
        elif any(dim in coord.dims for dim in interp_dims):
            interp_dict = {dim: target_ds.coords[dim] for dim in coord.dims if dim in interp_dims}
            new_coords[name] = coord.interp(interp_dict, method=method)
        else:
            new_coords[name] = coord

    # Re-attach coordinates and preserve attributes
    result = result.assign_coords(new_coords)
    if data.name:
        result.name = data.name

    return result


def _scipy_interp_wrapper(
    data: np.ndarray,
    src_coords: tuple[np.ndarray, ...],
    tgt_coords_1d: tuple[np.ndarray, ...],
    method: str,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Wrapper for RegularGridInterpolator to be used with apply_ufunc.

    Parameters
    ----------
    data : np.ndarray
        Input data slice. Core dimensions are at the end.
    src_coords : tuple[np.ndarray, ...]
        Source coordinate arrays.
    tgt_coords_1d : tuple[np.ndarray, ...]
        Target coordinate arrays.
    method : str
        Interpolation method.
    target_shape : tuple[int, ...]
        Desired output shape for the spatial dimensions.

    Returns
    -------
    np.ndarray
        Interpolated data slice. Core dimensions are at the end.
    """
    n_interp = len(src_coords)
    # RegularGridInterpolator expects core dimensions at the beginning.
    # apply_ufunc moves them to the end.
    if data.ndim > n_interp:
        axes = list(range(data.ndim))
        new_axes = axes[-n_interp:] + axes[:-n_interp]
        data_for_interp = data.transpose(new_axes)
    else:
        data_for_interp = data

    # Create interpolator
    interpolator = RegularGridInterpolator(src_coords, data_for_interp, method=method, bounds_error=False, fill_value=np.nan)

    # Generate target points grid
    tgt_mesh = np.meshgrid(*tgt_coords_1d, indexing="ij")
    flat_tgt = np.stack([m.ravel() for m in tgt_mesh], axis=-1)

    # Interpolate
    # Result has shape (N_target_flat, ...) where ... are extra dimensions
    new_values_flat = interpolator(flat_tgt)

    # Reshape to (target_shape, ...)
    extra_shape = data.shape[:-n_interp]
    new_values = new_values_flat.reshape(*target_shape, *extra_shape)

    # Move core dimensions back to the end for apply_ufunc
    if data.ndim > n_interp:
        axes = list(range(new_values.ndim))
        final_axes = axes[n_interp:] + axes[:n_interp]
        return new_values.transpose(final_axes)

    return new_values
