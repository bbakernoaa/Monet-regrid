"""Conservative regridding implementation."""

from __future__ import annotations

from collections.abc import Hashable
from typing import overload

import numpy as np
import xarray as xr

from monet_regrid import utils

try:
    import sparse
except ImportError:
    sparse = None

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
def conservative_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray:
    ...


@overload
def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.Dataset:
    ...


def conservative_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | Hashable | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using conservative regridding.

    The method implementation is based on a post by Stephan Hoyer:
    "For the case of interpolation between rectilinear grids (even on the sphere),
    you can factorize regridding along each axis. This is less general but makes
    the entire calculation much simpler, because its feasible to store
    interpolation weights as dense matrices and to use dense matrix multiplication."
    https://discourse.pangeo.io/t/conservative-region-aggregation-with-xarray-geopandas-and-sparse/2715

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input dataset to be regridded.
    target_ds : xr.Dataset
        Dataset containing the target coordinates.
    latitude_coord : str | Hashable | None
        Name of the latitude coordinate. If None, it will be inferred.
    skipna : bool, optional
        If True, handle NaN values during regridding. Defaults to True.
    nan_threshold : float, optional
        Threshold for keeping output points based on valid input points.
        Defaults to 1.0.
    output_chunks : dict[Hashable, int] | None, optional
        Explicit chunk sizes for the output data. Defaults to None.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Regridded input dataset.
    """
    # Attempt to infer the latitude coordinate
    if latitude_coord is None:
        for coord in data.coords:
            if str(coord).lower() in ["lat", "latitude"]:
                latitude_coord = coord
                break

    # Make sure the regridding coordinates are sorted
    coord_names = [coord for coord in target_ds.coords if coord in data.coords]
    target_ds_sorted = xr.Dataset(coords=target_ds.coords)
    for coord_name in coord_names:
        target_ds_sorted = utils.ensure_monotonic(target_ds_sorted, coord_name)
        data = utils.ensure_monotonic(data, coord_name)
    coords = {name: target_ds_sorted[name] for name in coord_names}

    regridded_data = utils.call_on_dataset(
        conservative_regrid_dataset,
        data,
        coords,
        latitude_coord,
        skipna,
        nan_threshold,
        output_chunks,
    )

    regridded_data = regridded_data.reindex_like(target_ds, copy=False)

    # Update history attribute for provenance
    utils.update_history(regridded_data, "Regridded using conservative method")

    return regridded_data


def conservative_regrid_dataset(
    data: xr.Dataset,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: Hashable | None,
    skipna: bool,
    nan_threshold: float,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.Dataset:
    """Dataset implementation of the conservative regridding method.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset to be regridded.
    coords : dict[Hashable, xr.DataArray]
        Dictionary of target coordinates.
    latitude_coord : Hashable | None
        Name of the latitude coordinate.
    skipna : bool
        If True, handle NaN values.
    nan_threshold : float
        Threshold for valid data.
    output_chunks : dict[Hashable, int] | None, optional
        Explicit chunk sizes for output. Defaults to None.

    Returns
    -------
    xr.Dataset
        The regridded dataset.
    """
    data_vars = dict(data.data_vars)
    data_coords = dict(data.coords)
    data_attrs = {v: data_vars[v].attrs for v in data_vars}
    coord_attrs = {c: data_coords[c].attrs for c in data_coords}
    ds_attrs = data.attrs

    # Create weights array and coverage mask for each regridding dim
    weights = {}
    covered = {}
    for coord, coord_array in coords.items():
        # Coverage check can be done lazily
        covered[coord] = (coord_array <= data[coord].max()) & (coord_array >= data[coord].min())

        # Weight calculation requires 1D coordinate values (usually small)
        # We compute them explicitly only here
        target_coords = coord_array.compute().data if not isinstance(coord_array.data, np.ndarray) else coord_array.data
        source_coords = data[coord].compute().data if not isinstance(data[coord].data, np.ndarray) else data[coord].data
        nd_weights = get_weights(source_coords, target_coords)

        da_weights = utils.create_dot_dataarray(nd_weights, str(coord), target_coords, source_coords)
        # Modify weights to correct for latitude distortion
        if coord == latitude_coord:
            da_weights = apply_spherical_correction(da_weights, latitude_coord)
        weights[coord] = da_weights

    # Apply the weights, using a unique set that matches chunking of each array
    for array, data_array in data_vars.items():
        var_weights = {}
        for coord, weight_array in weights.items():
            var_input_chunks = data_array.chunksizes.get(coord)
            var_output_chunks = output_chunks.get(coord) if output_chunks else None
            var_weights[coord] = format_weights(
                weight_array,
                coord,
                data_array.dtype,
                var_input_chunks,
                var_output_chunks,
            )

        data_vars[array] = apply_weights(
            da=data_array,
            weights=var_weights,
            skipna=skipna,
            nan_threshold=nan_threshold,
        )
        # Mask out any regridded points outside the original domain
        # Limit to dims present on this array otherwise .where broadcasts
        var_covered = xr.DataArray(True)
        for coord in var_weights.keys():
            var_covered = var_covered & covered[coord]
        data_vars[array] = data_vars[array].where(var_covered)

    # Rebuild the results ensuring we preserve attributes and other coordinates
    for array, attrs in data_attrs.items():
        data_vars[array].attrs = attrs

    ds_regridded = xr.Dataset(data_vars=data_vars, attrs=ds_attrs)

    for coord, attrs in coord_attrs.items():
        if coord not in ds_regridded.coords:
            # Add back any additional coordinates from the original dataset
            ds_regridded[coord] = data_coords[coord]
        ds_regridded[coord].attrs = attrs

    return ds_regridded


def apply_weights(
    da: xr.DataArray,
    weights: dict[Hashable, xr.DataArray],
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """Apply the weights over all regridding dimensions simultaneously with `xr.dot`.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray to be regridded.
    weights : dict[Hashable, xr.DataArray]
        Dictionary of weight matrices for each dimension.
    skipna : bool
        If True, handle NaN values.
    nan_threshold : float
        Threshold for valid data.

    Returns
    -------
    xr.DataArray
        The regridded DataArray.
    """
    coords = list(weights.keys())
    weight_arrays = list(weights.values())

    da_regrid: xr.DataArray = xr.dot(da.fillna(0), *weight_arrays, dim=list(weights.keys()), optimize=True)

    if skipna:
        valid_frac = xr.dot(da.notnull(), *weight_arrays, dim=list(weights.keys()), optimize=True)
        da_regrid /= valid_frac
        da_regrid = da_regrid.where(valid_frac >= get_valid_threshold(nan_threshold))

    # Rename temporary coordinates and ensure original dimension order
    coord_map = {f"target_{coord}": coord for coord in coords}
    da_regrid = da_regrid.rename(coord_map).transpose(*da.dims)

    return da_regrid


def get_valid_threshold(nan_threshold: float) -> float:
    """Invert the nan_threshold and coerce it to just above zero and below
    one to handle numerical precision limitations in the weight sum.

    Parameters
    ----------
    nan_threshold : float
        The threshold for NaN values (0 to 1).

    Returns
    -------
    float
        The adjusted threshold.
    """
    # This matches xesmf where na_thresh=0 keeps points with any valid data
    valid_threshold: float = 1 - np.clip(nan_threshold, 1e-6, 1.0 - 1e-6)
    return valid_threshold


def get_weights(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Determine the weights to map from the old coordinates to the new coordinates.

    Parameters
    ----------
    source_coords : np.ndarray
        Source coordinates (center points).
    target_coords : np.ndarray
        Target coordinates (center points).

    Returns
    -------
    np.ndarray
        Weights matrix of shape (len(source_coords), len(target_coords)).
    """
    target_intervals = utils.to_intervalindex(target_coords)
    source_intervals = utils.to_intervalindex(source_coords)

    overlap = utils.overlap(source_intervals, target_intervals)
    return utils.normalize_overlap(overlap)


def apply_spherical_correction(dot_array: xr.DataArray, latitude_coord: Hashable) -> xr.DataArray:
    """Apply a spherical earth correction on the prepared dot product weights.

    Parameters
    ----------
    dot_array : xr.DataArray
        The weight matrix for the latitude dimension.
    latitude_coord : Hashable
        Name of the latitude coordinate.

    Returns
    -------
    xr.DataArray
        Spherical-corrected weight matrix.
    """
    # Use xarray arithmetic for laziness
    lat_diff = dot_array[latitude_coord].diff(latitude_coord)
    # We use .median().compute().item() to get a scalar value for the resolution
    # This is a small computation on a 1D coordinate, so it's acceptable.
    latitude_res = float(lat_diff.median().compute().item())

    # Calculate weights using vectorized xarray arithmetic
    da_lat_weights = lat_weight_xarray(dot_array[latitude_coord], latitude_res)

    # Use xarray arithmetic to maintain potential laziness/provenance
    corrected_weights = dot_array * da_lat_weights

    # Normalize along the source latitude dimension (latitude_coord) to ensure
    # that each target cell's weights sum to 1.0.
    return corrected_weights / corrected_weights.sum(dim=latitude_coord)


def lat_weight_xarray(latitude: xr.DataArray, latitude_res: float) -> xr.DataArray:
    """Return the weight of gridcells based on their latitude using Xarray.

    Parameters
    ----------
    latitude : xr.DataArray
        (Center) latitude values of the gridcells, in degrees.
    latitude_res : float
        Resolution/width of the grid cells, in degrees.

    Returns
    -------
    xr.DataArray
        Weights, same shape as latitude input.
    """
    dlat: float = np.radians(latitude_res)
    lat_rad = np.radians(latitude)
    h = np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2)
    return h * dlat / (np.pi * 4)


def lat_weight(latitude: np.ndarray, latitude_res: float) -> np.ndarray:
    """Return the weight of gridcells based on their latitude.

    Parameters
    ----------
    latitude : np.ndarray
        (Center) latitude values of the gridcells, in degrees.
    latitude_res : float
        Resolution/width of the grid cells, in degrees.

    Returns
    -------
    np.ndarray
        Weights, same shape as latitude input.
    """
    dlat: float = np.radians(latitude_res)
    lat = np.radians(latitude)
    h = np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)
    return h * dlat / (np.pi * 4)


def format_weights(
    weights: xr.DataArray,
    coord: Hashable,
    input_dtype: np.dtype,
    input_chunks: tuple[int, ...] | None,
    output_chunks: tuple[int, ...] | int | None,
) -> xr.DataArray:
    """Format the raw weights array to match input properties and optimize performance.

    Ensures weights match the input dtype, follows a 1:1 chunking strategy with
    source data, and converts to a sparse representation if the `sparse` package
    is available.

    Parameters
    ----------
    weights : xr.DataArray
        The raw weights DataArray.
    coord : Hashable
        The coordinate name.
    input_dtype : np.dtype
        The dtype of the input data to match.
    input_chunks : tuple[int, ...] | None
        Chunks of the input data.
    output_chunks : tuple[int, ...] | int | None
        Desired chunks for the output data.

    Returns
    -------
    xr.DataArray
        The formatted weights DataArray, potentially sparse and dask-backed.
    """
    # Use single precision weights at minimum, double if input is double
    weights_dtype = np.result_type(np.float32, input_dtype)
    new_weights = weights.copy().astype(weights_dtype)

    chunks: dict[Hashable, tuple[int, ...] | int] = {}
    if input_chunks is not None:
        chunks[coord] = input_chunks
        if output_chunks is None:
            # Set output chunking to match input, but precise chunks won't match shape,
            # so take the max in case of uneven chunks
            output_chunks = max(input_chunks)

    if output_chunks is not None:
        chunks[f"target_{coord}"] = output_chunks

    if chunks:
        new_weights = new_weights.chunk(chunks)
        if sparse is not None:
            # Use a safe sparse conversion that doesn't hang
            # We wrap sparse.COO in a way that dask handles correctly
            new_weights.data = new_weights.data.map_blocks(
                lambda x: sparse.COO.from_numpy(x) if isinstance(x, np.ndarray) else x, dtype=new_weights.dtype
            )
    elif sparse is not None:
        new_weights.data = sparse.COO.from_numpy(weights.data)

    return new_weights
