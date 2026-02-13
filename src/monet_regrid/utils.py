from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypedDict, overload

import cf_xarray  # noqa: F401
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from monet_regrid.constants import GridType

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


class InvalidBoundsError(Exception):
    ...


class CoordHandler(TypedDict):
    names: list[str]
    func: Callable


@dataclass
class Grid:
    """Object storing grid information."""

    north: float
    east: float
    south: float
    west: float
    resolution_lat: float
    resolution_lon: float

    def __post_init__(self) -> None:
        """Validate the initialized SpatialBounds class."""
        msg = None
        if self.south > self.north:
            msg = "Value of north bound is greater than south bound.\nPlease check the bounds input."
            pass
        if self.west > self.east:
            msg = "Value of west bound is greater than east bound.\nPlease check the bounds input."
        if msg is not None:
            raise InvalidBoundsError(msg)

    def create_regridding_dataset(self, lat_name: str = "latitude", lon_name: str = "longitude") -> xr.Dataset:
        """Create a dataset to use for regridding.

        Parameters
        ----------
        lat_name : str, optional
            Name for the latitudinal coordinate and dimension.
            Defaults to "latitude".
        lon_name : str, optional
            Name for the longitudinal coordinate and dimension.
            Defaults to "longitude".

        Returns
        -------
        xr.Dataset
            A dataset with the latitude and longitude coordinates corresponding to the
            specified grid. Contains no data variables.
        """
        return create_regridding_dataset(self, lat_name, lon_name)


def create_lat_lon_coords(
    grid: Grid,
    chunks: int | dict[str, int] | str | None = None,
) -> tuple[da.Array, da.Array]:
    """Create lazily-computed latitude and longitude coordinates.

    This function uses Dask to generate coordinate arrays, preventing them from
    being loaded into memory until they are explicitly computed. This is critical
    for handling large, high-resolution grids efficiently.

    Parameters
    ----------
    grid : Grid
        A `Grid` object containing the spatial bounds and resolution.
    chunks : int | dict[str, int] | str | None, optional
        Chunk size for the Dask arrays. If None, Dask will choose a default.
        Recommended to use a size that results in ~100MB chunks for large grids.

    Returns
    -------
    tuple[da.Array, da.Array]
        A tuple containing the lazily-computed latitude and longitude Dask arrays.
    """
    arange_kwargs = {}
    if chunks is not None:
        arange_kwargs["chunks"] = chunks

    if np.remainder((grid.north - grid.south), grid.resolution_lat) > 0:
        lat_coords = da.arange(grid.south, grid.north, grid.resolution_lat, **arange_kwargs)
    else:
        lat_coords = da.arange(grid.south, grid.north + grid.resolution_lat, grid.resolution_lat, **arange_kwargs)

    if np.remainder((grid.east - grid.west), grid.resolution_lon) > 0:
        lon_coords = da.arange(grid.west, grid.east, grid.resolution_lon, **arange_kwargs)
    else:
        lon_coords = da.arange(grid.west, grid.east + grid.resolution_lon, grid.resolution_lon, **arange_kwargs)

    return lat_coords, lon_coords


def create_regridding_dataset(
    grid: Grid,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    chunks: int | dict[str, int] | str | None = None,
) -> xr.Dataset:
    """Create a lazy xarray.Dataset for regridding.

    This function constructs a dataset containing only coordinate information,
    which is generated lazily using Dask. This approach is highly memory-efficient
    for defining large target grids for regridding operations.

    Parameters
    ----------
    grid : Grid
        A `Grid` object defining the spatial bounds and resolution.
    lat_name : str, optional
        The desired name for the latitude coordinate, by default "latitude".
    lon_name : str, optional
        The desired name for the longitude coordinate, by default "longitude".
    chunks : int | dict[str, int] | str | None, optional
        Chunk size for the coordinate arrays.

    Returns
    -------
    xr.Dataset
        A dataset with Dask-backed latitude and longitude coordinates and no
        data variables.
    """
    lat_coords, lon_coords = create_lat_lon_coords(grid, chunks=chunks)
    return xr.Dataset(
        coords={
            lat_name: ([lat_name], lat_coords, {"units": "degrees_north"}),
            lon_name: ([lon_name], lon_coords, {"units": "degrees_east"}),
        }
    )


def to_intervalindex(coords: np.ndarray) -> pd.IntervalIndex:
    """Convert a 1-d coordinate array to a pandas IntervalIndex.

    The midpoints between the coordinates are taken as the interval boundaries.

    Parameters
    ----------
    coords : np.ndarray
        1-d array containing the coordinate values.

    Returns
    -------
    pd.IntervalIndex
        A pandas IntervalIndex containing the intervals corresponding to the input
        coordinates.
    """
    if len(coords) > 1:
        midpoints = (coords[:-1] + coords[1:]) / 2

        # Extrapolate outer bounds beyond the first and last coordinates
        left_bound = 2 * coords[0] - midpoints[0]
        right_bound = 2 * coords[-1] - midpoints[-1]

        breaks = np.concatenate([[left_bound], midpoints, [right_bound]])
        intervals = pd.IntervalIndex.from_breaks(breaks)

    else:
        # If the target grid has a single point, set search interval to span all space
        intervals = pd.IntervalIndex.from_breaks([-np.inf, np.inf])

    return intervals


def overlap(a: pd.IntervalIndex, b: pd.IntervalIndex) -> np.ndarray:
    """Calculate the overlap between two sets of intervals.

    The overlap is calculated in a vectorized manner using NumPy broadcasting.
    To optimize performance, the broadcasting order is chosen based on the relative
    sizes of the two interval sets.

    Parameters
    ----------
    a : pd.IntervalIndex
        The first set of intervals (typically source coordinates).
    b : pd.IntervalIndex
        The second set of intervals (typically target coordinates).

    Returns
    -------
    np.ndarray
        A 2D array of shape (len(a), len(b)) containing the overlap length
        between each pair of intervals.
    """
    if len(a) >= len(b):
        mins = np.minimum(a.right.to_numpy(), b.right.to_numpy()[:, np.newaxis])
        maxs = np.maximum(a.left.to_numpy(), b.left.to_numpy()[:, np.newaxis])
        overlap_vals: np.ndarray = np.maximum(mins - maxs, 0).T
    else:
        mins = np.minimum(a.right.to_numpy()[:, np.newaxis], b.right.to_numpy())
        maxs = np.maximum(a.left.to_numpy()[:, np.newaxis], b.left.to_numpy())
        overlap_vals = np.maximum(mins - maxs, 0)

    return overlap_vals


def normalize_overlap(overlap: np.ndarray) -> np.ndarray:
    """Normalize overlap values so they sum up to 1.0 along the first axis.

    Parameters
    ----------
    overlap : np.ndarray
        2D array of overlap values.

    Returns
    -------
    np.ndarray
        The normalized overlap values.
    """
    overlap_sum: np.ndarray = overlap.sum(axis=0)
    overlap_sum[overlap_sum == 0] = 1e-12  # Avoid dividing by 0.
    return overlap / overlap_sum


def create_dot_dataarray(
    weights: np.ndarray,
    coord: str,
    target_coords: np.ndarray,
    source_coords: np.ndarray,
) -> xr.DataArray:
    """Create a DataArray to be used at dot product compatible with xr.dot.

    Parameters
    ----------
    weights : np.ndarray
        The regridding weights.
    coord : str
        The name of the coordinate being regridded.
    target_coords : np.ndarray
        The target coordinate values.
    source_coords : np.ndarray
        The source coordinate values.

    Returns
    -------
    xr.DataArray
        A DataArray containing the weights and appropriate coordinates.
    """
    return xr.DataArray(
        data=weights,
        dims=[coord, f"target_{coord}"],
        coords={
            coord: source_coords,
            f"target_{coord}": target_coords,
        },
    )


def common_coords(
    data1: xr.DataArray | xr.Dataset,
    data2: xr.DataArray | xr.Dataset,
    remove_coord: str | None = None,
) -> list[Hashable]:
    """Return a set of coords which two dataset/arrays have in common.

    Parameters
    ----------
    data1 : xr.DataArray | xr.Dataset
        First xarray object.
    data2 : xr.DataArray | xr.Dataset
        Second xarray object.
    remove_coord : str | None, optional
        A coordinate name to exclude from the result.

    Returns
    -------
    list[Hashable]
        List of common coordinate names.
    """
    coords = set(data1.coords).intersection(set(data2.coords))
    if remove_coord in coords:
        coords.remove(remove_coord)
    return list(coords)


def call_on_dataset(
    func: Callable[..., xr.Dataset],
    obj: xr.DataArray | xr.Dataset,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Call a function that expects a Dataset on either a Dataset or DataArray.

    This utility handles the round-tripping between DataArray and Dataset.

    Parameters
    ----------
    func : Callable[..., xr.Dataset]
        The function to be called.
    obj : xr.DataArray | xr.Dataset
        The input object.
    *args : Any
        Positional arguments for `func`.
    **kwargs : Any
        Keyword arguments for `func`.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The result of `func`, cast back to the original type of `obj`.
    """
    placeholder_name = "_UNNAMED_ARRAY"
    if isinstance(obj, xr.DataArray):
        tmp_name = obj.name if obj.name is not None else placeholder_name
        ds = obj.to_dataset(name=tmp_name)
    else:
        ds = obj

    result = func(ds, *args, **kwargs)

    if isinstance(obj, xr.DataArray) and isinstance(result, xr.Dataset):
        msg = "Trying to convert Dataset with more than one data variable to DataArray"
        if len(result.data_vars) > 1:
            raise TypeError(msg)
        return next(iter(result.data_vars.values())).rename(obj.name)

    return result


@overload
def format_for_regrid(
    obj: xr.Dataset,
    target: xr.Dataset,
    stats: bool = False,
) -> xr.Dataset:
    ...


@overload
def format_for_regrid(
    obj: xr.DataArray,
    target: xr.Dataset,
    stats: bool = False,
) -> xr.DataArray:
    ...


def format_for_regrid(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    stats: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Apply pre-formatting to the input dataset to prepare for regridding.

    Handles padding of spherical geometry if lat/lon coordinates can be inferred
    and the domain size requires boundary padding. This ensures that the
    regridding process has full coverage for the target grid.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The source data to be formatted.
    target : xr.Dataset
        The target grid dataset.
    stats : bool, optional
        If True, skip latitude padding as it can alter statistical aggregations.
        Defaults to False.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The formatted data, potentially with added pole padding or shifted
        longitude coordinates.
    """
    # Special-cased coordinates with accepted names and formatting function
    coord_handlers: dict[str, CoordHandler] = {
        "lat": {"names": ["lat", "latitude"], "func": format_lat},
        "lon": {"names": ["lon", "longitude"], "func": format_lon},
    }

    # Latitude padding adds a duplicate value which will undesirably
    # alter statistical aggregations
    if stats:
        coord_handlers.pop("lat")

    # Identify coordinates that need to be formatted
    formatted_coords = {}
    target_coords = {}  # Map coord_type to target coordinate name

    for coord_type, handler in coord_handlers.items():
        # Find source coordinate
        for coord in obj.coords.keys():
            if str(coord).lower() in handler["names"]:
                formatted_coords[coord_type] = str(coord)
                break

        # Find corresponding target coordinate
        for target_coord in target.coords.keys():
            if str(target_coord).lower() in handler["names"]:
                target_coords[coord_type] = str(target_coord)
                break

    # Apply formatting
    result = obj.copy()
    for coord_type, coord in formatted_coords.items():
        # Make sure formatted coords are sorted
        result = ensure_monotonic(result, coord)

        # For target, use the target coordinate name if available, otherwise skip
        target_coord = target_coords.get(coord_type)
        if target_coord:
            target = ensure_monotonic(target, target_coord)

        result = coord_handlers[coord_type]["func"](result, target, formatted_coords)

        # Coerce back to a single chunk if that's what was passed
        if isinstance(obj, xr.DataArray) and len(obj.chunksizes.get(coord, ())) == 1:
            result = result.chunk({coord: -1})
        elif isinstance(obj, xr.Dataset):
            for var in result.data_vars:
                if len(obj[var].chunksizes.get(coord, ())) == 1:
                    result[var] = result[var].chunk({coord: -1})

    # Update history attribute for provenance
    update_history(result, "Pre-formatted data for regridding (pole padding/longitude shift)")

    return result


def format_lat(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,  # noqa ARG001
    formatted_coords: dict[str, str],
) -> xr.DataArray | xr.Dataset:
    """Apply pole padding to the latitude coordinate if it is inferred to be global.

    If the latitude range is nearly global but does not include the poles, this
    function adds values at -90 and 90 degrees. The values at the poles are
    computed as the mean of the first and last latitude bands, respectively.
    This is equivalent to the `Pole="all"` option in `ESMF`.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The source data to be formatted.
    target : xr.Dataset
        The target grid dataset.
    formatted_coords : dict[str, str]
        Dictionary mapping coordinate types (e.g., 'lat') to their names in `obj`.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The data with potential pole padding applied.

    Notes
    -----
    Pole padding is only applied to 1D latitude coordinates (rectilinear grids).
    For 2D coordinates (curvilinear grids), no padding is performed.
    """
    lat_coord = formatted_coords["lat"]
    lon_coord = formatted_coords.get("lon")

    # Check if this is a 2D coordinate (curvilinear grid)
    if obj.coords[lat_coord].data.ndim == 2:
        # For curvilinear grids, skip pole padding
        return obj

    # Concat a padded value representing the mean of the first/last lat bands
    # This should match the Pole="all" option of ESMF
    # TODO: with cos(90) = 0 weighting, these weights might be 0?

    polar_lat = 90
    # Use max() on diff to get resolution. item() is acceptable for scalar result.
    dy: Any = float(obj.coords[lat_coord].diff(lat_coord).max().compute())

    # Only pad if global but don't have edge values directly at poles
    # NOTE: could use xr.pad here instead of xr.concat, but none of the
    # modes are an exact fit for this scheme

    lat_dim = obj[lat_coord].dims[0]
    lon_dim = obj[lon_coord].dims[0] if lon_coord else None

    lat_coord_da = obj.coords[lat_coord]

    # South pole - Use lazy checks where possible
    first_lat = float(lat_coord_da.isel({lat_dim: 0}).compute())
    if dy - polar_lat >= first_lat > -polar_lat:
        south_pole = obj.isel({lat_dim: 0})
        if lon_dim is not None:
            south_pole = south_pole.mean(lon_dim, keep_attrs=True)
        attrs = obj.coords[lat_coord].attrs
        obj = xr.concat([south_pole, obj], dim=lat_dim)  # type: ignore
        # Update coordinate values lazily
        new_val = np.array([-polar_lat], dtype=lat_coord_da.dtype)
        if hasattr(lat_coord_da.data, "chunks"):
            new_val = da.from_array(new_val, chunks=1)
            new_lat_data = da.concatenate([new_val, lat_coord_da.data], axis=0)
        else:
            new_lat_data = np.concatenate([new_val, lat_coord_da.values], axis=0)

        obj = obj.assign_coords({lat_coord: (lat_dim, new_lat_data)})
        obj.coords[lat_coord].attrs = attrs
        lat_coord_da = obj.coords[lat_coord]

    # North pole
    last_lat = float(lat_coord_da.isel({lat_dim: -1}).compute())
    if polar_lat - dy <= last_lat < polar_lat:
        attrs = obj.coords[lat_coord].attrs
        north_pole = obj.isel({lat_dim: -1})
        if lon_dim is not None:
            north_pole = north_pole.mean(lon_dim, keep_attrs=True)
        obj = xr.concat([obj, north_pole], dim=lat_dim)  # type: ignore
        # Update coordinate values lazily
        new_val = np.array([polar_lat], dtype=lat_coord_da.dtype)
        if hasattr(lat_coord_da.data, "chunks"):
            new_val = da.from_array(new_val, chunks=1)
            new_lat_data = da.concatenate([lat_coord_da.data, new_val], axis=0)
        else:
            new_lat_data = np.concatenate([lat_coord_da.values, new_val], axis=0)

        obj = obj.assign_coords({lat_coord: (lat_dim, new_lat_data)})
        obj.coords[lat_coord].attrs = attrs

    return obj


def format_lon(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    formatted_coords: dict[str, str],
) -> xr.DataArray | xr.Dataset:
    """Format the longitude coordinate to align with the target grid.

    Shifts the source grid longitude to match the target range and adds
    wraparound padding if the domain is global and the target extends beyond
    the source centers.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The source data to be formatted.
    target : xr.Dataset
        The target grid dataset.
    formatted_coords : dict[str, str]
        Dictionary mapping coordinate types (e.g., 'lon') to their names in `obj`.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The data with formatted longitude coordinates.

    Notes
    -----
    Longitude formatting is only applied to 1D coordinates (rectilinear grids).
    """
    lon_coord = formatted_coords["lon"]

    # Check if this is a 2D coordinate (curvilinear grid)
    if obj.coords[lon_coord].data.ndim == 2:
        # For curvilinear grids, skip longitude formatting
        return obj

    # Find the corresponding longitude coordinate in the target dataset
    target_lon_coord = None
    for coord_name in target.coords:
        if str(coord_name).lower() in ["lon", "longitude"]:
            target_lon_coord = coord_name
            break

    # If we can't find a target longitude coordinate, skip the formatting
    if target_lon_coord is None:
        return obj

    # Find a wrap point outside of the left and right bounds of the target
    # This ensures we have coverage on the target and handles global > regional
    source_lon = obj.coords[lon_coord]
    target_lon_da = target.coords[target_lon_coord]

    # Use compute() for scalars needed for logic
    t_first = float(target_lon_da.isel({target_lon_da.dims[0]: 0}).compute())
    t_last = float(target_lon_da.isel({target_lon_da.dims[0]: -1}).compute())
    wrap_point = (t_last + t_first + 360) / 2

    # Use xr.where for laziness
    new_source_lon = xr.where(source_lon < wrap_point - 360, source_lon + 360, source_lon)
    new_source_lon = xr.where(new_source_lon > wrap_point, new_source_lon - 360, new_source_lon)
    obj = obj.assign_coords({lon_coord: new_source_lon})

    obj = ensure_monotonic(obj, lon_coord)

    # Only pad if domain is global in lon
    source_lon = obj.coords[lon_coord]
    dx_s: Any = float(source_lon.diff(lon_coord).max().compute())
    dx_t: Any = float(target_lon_da.diff(target_lon_da.dims[0]).max().compute())
    is_global_lon = bool((source_lon.max() - source_lon.min()).compute() >= 360 - dx_s)

    if is_global_lon:
        s_first = float(source_lon.isel({lon_coord: 0}).compute())
        s_last = float(source_lon.isel({lon_coord: -1}).compute())

        left_pad = int(np.ceil(max((s_first - t_first + dx_t / 2) / dx_s, 0)))
        right_pad = int(np.ceil(max((t_last - s_last + dx_t / 2) / dx_s, 0)))

        if left_pad > 0 or right_pad > 0:
            attrs = obj.coords[lon_coord].attrs
            obj = obj.pad({lon_coord: (left_pad, right_pad)}, mode="wrap", keep_attrs=True)

            # Update coordinate values lazily using xr.where
            lon_da = obj.coords[lon_coord]
            n_total = lon_da.sizes[lon_coord]
            # Create an index array for masking
            indices = xr.DataArray(np.arange(n_total), dims=[lon_coord])
            if hasattr(lon_da.data, "chunks"):
                indices = indices.chunk(lon_da.chunks)

            new_lon = xr.where(indices < left_pad, lon_da - 360, lon_da)
            new_lon = xr.where(indices >= n_total - right_pad, new_lon + 360, new_lon)

            obj = obj.assign_coords({lon_coord: new_lon})
            obj.coords[lon_coord].attrs = attrs
            obj = ensure_monotonic(obj, lon_coord)

    return obj


def coord_is_covered(obj: xr.DataArray | xr.Dataset, target: xr.Dataset, coord: Hashable) -> bool:
    """Check if the source coordinate fully covers the target coordinate.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The source data object.
    target : xr.Dataset
        The target grid dataset.
    coord : Hashable
        The name of the coordinate to check.

    Returns
    -------
    bool
        True if the source coordinate covers the target coordinate range,
        False otherwise.
    """
    pad = target[coord].diff(coord).max().values
    left_covered = obj[coord].min() <= target[coord].min() - pad
    right_covered = obj[coord].max() >= target[coord].max() + pad
    return bool(left_covered.item() and right_covered.item())


@overload
def ensure_monotonic(obj: xr.DataArray, coord: Hashable) -> xr.DataArray:
    ...


@overload
def ensure_monotonic(obj: xr.Dataset, coord: Hashable) -> xr.Dataset:
    ...


def ensure_monotonic(obj: xr.DataArray | xr.Dataset, coord: Hashable) -> xr.DataArray | xr.Dataset:
    """Ensure that an object has monotonically increasing indexes for a
    given coordinate. Only sort and drop duplicates if needed because this
    requires reindexing which can be expensive."""
    # Check if the coordinate is actually an indexed dimension coordinate
    if coord in obj.indexes and coord in obj.dims:
        if not obj.indexes[coord].is_monotonic_increasing:
            obj = obj.sortby(coord)
        if not obj.indexes[coord].is_unique:
            obj = obj.drop_duplicates(coord)
    return obj


@overload
def update_coord(obj: xr.DataArray, coord: Hashable, coord_vals: np.ndarray) -> xr.DataArray:
    ...


@overload
def update_coord(obj: xr.Dataset, coord: Hashable, coord_vals: np.ndarray) -> xr.Dataset:
    ...


def update_coord(obj: xr.DataArray | xr.Dataset, coord: Hashable, coord_vals: np.ndarray) -> xr.DataArray | xr.Dataset:
    """Update the values of a coordinate, ensuring indexes stay in sync."""
    attrs = obj.coords[coord].attrs
    dims = obj.coords[coord].dims
    obj = obj.assign_coords({coord: (dims, coord_vals)})
    obj.coords[coord].attrs = attrs
    return obj


def _get_grid_type(ds: xr.Dataset) -> GridType:
    """Detect the grid type of the dataset using cf-xarray.

    Uses cf-xarray to access coordinate information and determine if the grid
    is rectilinear (1D coordinates) or curvilinear (2D coordinates).

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset to inspect.

    Returns
    -------
    GridType
        Either GridType.RECTILINEAR or GridType.CURVILINEAR.

    Raises
    ------
    ValueError
        If coordinates cannot be identified or if mixed dimensions are found.
    """
    try:
        # Import cf_xarray to ensure it's registered with xarray

        # Access latitude and longitude coordinates using cf-xarray
        try:
            lat_coord = ds.cf["latitude"]
            lon_coord = ds.cf["longitude"]

            # Check the number of dimensions for each coordinate
            lat_ndim = lat_coord.ndim
            lon_ndim = lon_coord.ndim

            # Both coordinates should have the same number of dimensions
            if lat_ndim != lon_ndim:
                msg = f"Mismatched coordinate dimensions: latitude has {lat_ndim} dims, longitude has {lon_ndim} dims"
                raise ValueError(msg) from None

            # Determine grid type based on dimensionality
            if lat_ndim == 1:
                return GridType.RECTILINEAR
            elif lat_ndim == 2:
                # Any 2D coordinates are treated as curvilinear since
                # rectilinear interpolation expects 1D dimension coordinates
                return GridType.CURVILINEAR
            else:
                msg = f"Unsupported coordinate dimensions: {lat_ndim} (expected 1 or 2)"
                raise ValueError(msg) from None

        except KeyError:
            # Fallback to manual search for coordinate names and check their dimensions
            # Look for coordinates that represent latitude/longitude regardless of name
            lat_coord_names = [name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["lat", "yc", "y"])]
            lon_coord_names = [name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["lon", "xc", "x"])]

            # If we have both lat and lon coordinates, check their dimensions
            if lat_coord_names and lon_coord_names:
                lat_coord = ds[lat_coord_names[0]]
                lon_coord = ds[lon_coord_names[0]]

                # Check the number of dimensions for each coordinate
                lat_ndim = lat_coord.ndim
                lon_ndim = lon_coord.ndim

                # Both coordinates should have the same number of dimensions
                if lat_ndim != lon_ndim:
                    msg = f"Mismatched coordinate dimensions: latitude has {lat_ndim} dims, longitude has {lon_ndim} dims"
                    raise ValueError(msg) from None

                # Determine grid type based on dimensionality
                if lat_ndim == 1:
                    return GridType.RECTILINEAR
                elif lat_ndim == 2:
                    # Any 2D coordinates are treated as curvilinear since
                    # rectilinear interpolation expects 1D dimension coordinates.
                    # We avoid eager .values calls here to maintain laziness.
                    return GridType.CURVILINEAR
                else:
                    msg = f"Unsupported coordinate dimensions: {lat_ndim} (expected 1 or 2)"
                    raise ValueError(msg) from None
            else:
                # If we don't have lat/lon or x/y named coordinates, try to identify coordinates by their dimensions
                # Look for coordinates that have 2 dimensions (which would indicate curvilinear)
                potential_lat_coords = []
                potential_lon_coords = []

                for coord_name in ds.coords:
                    coord_var = ds[coord_name]
                    if coord_var.ndim == 2:
                        # If we find 2D coordinates, check if they look like latitude/longitude
                        if "lat" in str(coord_name).lower() or "yc" in str(coord_name).lower() or "y" in str(coord_name).lower():
                            potential_lat_coords.append(coord_name)
                        elif "lon" in str(coord_name).lower() or "xc" in str(coord_name).lower() or "x" in str(coord_name).lower():
                            potential_lon_coords.append(coord_name)

                # If we still don't have any lat/lon coords, look for any 2D coordinates
                # and try to determine if they represent latitude/longitude based on units
                if not potential_lat_coords and not potential_lon_coords:
                    for coord_name in ds.coords:
                        coord_var = ds[coord_name]
                        if coord_var.ndim == 2:
                            attrs = coord_var.attrs
                            units = attrs.get("units", "").lower()
                            if "degree" in units and ("north" in units or "lat" in str(coord_name).lower()):
                                potential_lat_coords.append(coord_name)
                            elif "degree" in units and ("east" in units or "lon" in str(coord_name).lower()):
                                potential_lon_coords.append(coord_name)

                # If we have found potential lat/lon coordinates
                if potential_lat_coords and potential_lon_coords:
                    # Use the first ones found
                    lat_coord = ds[potential_lat_coords[0]]
                    lon_coord = ds[potential_lon_coords[0]]

                    # Both should be 2D for curvilinear
                    if lat_coord.ndim == 2 and lon_coord.ndim == 2:
                        return GridType.CURVILINEAR
                    else:
                        msg = f"Coordinates found but not both 2D: lat={lat_coord.ndim}D, lon={lon_coord.ndim}D"
                        raise ValueError(msg) from None
                else:
                    # If we still can't find them, try to find any 1D coordinates that look like lat/lon
                    for coord_name in ds.coords:
                        coord_var = ds[coord_name]
                        if coord_var.ndim == 1:
                            if (
                                "lat" in str(coord_name).lower()
                                or "lon" in str(coord_name).lower()
                                or "y" in str(coord_name).lower()
                                or "x" in str(coord_name).lower()
                            ):
                                # This is likely a rectilinear grid with 1D coordinates
                                # Look for both lat and lon
                                lat_1d_names = [
                                    name
                                    for name in ds.coords
                                    if ds[name].ndim == 1 and ("lat" in str(name).lower() or "y" in str(name).lower())
                                ]
                                lon_1d_names = [
                                    name
                                    for name in ds.coords
                                    if ds[name].ndim == 1 and ("lon" in str(name).lower() or "x" in str(name).lower())
                                ]

                                if lat_1d_names and lon_1d_names:
                                    return GridType.RECTILINEAR

                # If no coordinates found, check if it has dimensions
                if len(ds.dims) >= 2:
                    # Default to RECTILINEAR if we have at least 2 dimensions
                    # but no specific coordinates identified. This allows
                    # regridding of raw arrays by generating coordinates from dims.
                    return GridType.RECTILINEAR

                msg = "No latitude or longitude coordinates found"
                raise ValueError(msg) from None

    except (KeyError, ValueError) as e:
        msg = f"Could not identify coordinate: {e}"
        raise ValueError(msg) from e
    except AttributeError as e:
        # cf-xarray might not be available or coordinates not properly defined
        msg = "cf-xarray coordinate detection failed - coordinates not properly defined"
        raise ValueError(msg) from e


def validate_grid_compatibility(source_ds: xr.Dataset, target_ds: xr.Dataset) -> tuple[GridType, GridType]:
    """Validate that both source and target grids have supported types.

    Parameters
    ----------
    source_ds : xr.Dataset
        Source dataset.
    target_ds : xr.Dataset
        Target dataset.

    Returns
    -------
    tuple[GridType, GridType]
        Tuple of (source_grid_type, target_grid_type).

    Raises
    ------
    ValueError
        If either grid has an unsupported type.
    """
    source_type = _get_grid_type(source_ds)
    target_type = _get_grid_type(target_ds)

    # Currently only support rectilinear and curvilinear grids
    if source_type not in [GridType.RECTILINEAR, GridType.CURVILINEAR]:
        msg = f"Unsupported source grid type: {source_type}"
        raise ValueError(msg)

    if target_type not in [GridType.RECTILINEAR, GridType.CURVILINEAR]:
        msg = f"Unsupported target grid type: {target_type}"
        raise ValueError(msg)

    return source_type, target_type


@overload
def validate_input(
    data: xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,
) -> xr.Dataset:
    ...


@overload
def validate_input(
    data: xr.DataArray,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,
) -> xr.Dataset:
    ...


def validate_input(
    data: xr.DataArray | xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,
) -> xr.Dataset:
    """Validate and prepare the target grid for regridding.

    This function identifies the spatial coordinates in the source and target
    grids and constructs a new, validated target grid. It ensures that any
    non-spatial coordinates (like 'time') from the original target grid are
    preserved.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The source DataArray or Dataset.
    ds_target_grid : xr.Dataset
        The target grid Dataset.
    time_dim : str | None
        The name of the time dimension, if any.

    Returns
    -------
    xr.Dataset
        A new xr.Dataset representing the validated target grid.
    """
    _, _ = identify_cf_coordinates(data)  # Fails early if source coords are missing
    target_lat, target_lon = identify_cf_coordinates(ds_target_grid)

    validated_coords = {
        target_lat: ds_target_grid[target_lat],
        target_lon: ds_target_grid[target_lon],
    }

    # Add all other coordinates from the target grid that are not the identified
    # spatial coordinates. This preserves dimensions like 'time'.
    for coord_name, coord_da in ds_target_grid.coords.items():
        if coord_name not in [target_lat, target_lon]:
            validated_coords[coord_name] = coord_da

    # Ensure the specified time_dim is included if it exists. This is slightly
    # redundant with the loop above but acts as a safeguard.
    if time_dim and time_dim in ds_target_grid.coords:
        validated_coords[time_dim] = ds_target_grid[time_dim]

    return xr.Dataset(coords=validated_coords)


def _create_cache_key(data: xr.DataArray | xr.Dataset, time_dim: str | None = None) -> tuple:
    """Create a stable cache key from an xarray object's metadata.

    This key is based on the coordinates and dimensions, making it suitable for
    caching operations that depend on the grid structure rather than the data values.
    It uses a sampling strategy for large coordinates to avoid breaking laziness.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The xarray DataArray or Dataset.
    time_dim : str | None, optional
        The name of the time dimension, if any. Defaults to None.

    Returns
    -------
    tuple
        A hashable tuple that serves as a cache key.
    """
    coord_infos = []
    for name, coord in data.coords.items():
        # For small coordinates, using tobytes() is safe and very accurate
        # For large coordinates, it triggers computation and is slow
        if coord.size < 1000:
            coord_val = coord.values.tobytes()
        else:
            # Use sampling for large coordinates to maintain laziness
            # We take a few points from the start, middle, and end
            n = coord.size
            if coord.chunks is not None:
                # If dask-backed, use a strategy that doesn't trigger full compute
                indices = [0, n // 2, n - 1]
                sample = []
                # Flatten the data array to 1D for consistent indexing
                flat_coord = coord.data.ravel()
                for idx in indices:
                    # Use a small slice and compute only that
                    sample.append(float(flat_coord[idx : idx + 1].compute()[0]))
                coord_val = tuple(sample)
            else:
                # If numpy-backed, we can sample easily
                flat_vals = coord.values.ravel()
                indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
                coord_val = tuple(float(flat_vals[idx]) for idx in indices)

        coord_infos.append((name, coord.shape, coord.dtype, coord_val))

    coords_key = frozenset(coord_infos)
    dims_key = tuple(sorted(data.dims))

    return (coords_key, dims_key, time_dim)


def update_history(obj: xr.DataArray | xr.Dataset, message: str) -> None:
    """Update the history attribute of an xarray object for provenance tracking.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
        The xarray object to update.
    message : str
        The message to append to the history.

    Returns
    -------
    None
        The object is modified in-place.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 2, 3], attrs={"history": "Original"})
    >>> update_history(da, "Interpolated using monet_regrid")
    >>> print(da.attrs["history"])
    Original
    Interpolated using monet_regrid
    """
    existing_history = obj.attrs.get("history", "")
    obj.attrs["history"] = f"{existing_history}\n{message}" if existing_history else message


def identify_cf_coordinates(ds: xr.Dataset) -> tuple[str, str]:
    """Identify latitude and longitude coordinates using a fallback strategy.

    This function attempts to find the names of the latitude and longitude
    coordinates in a given xarray Dataset. It uses a hybrid strategy:

    1.  **CF-xarray:** First, it tries to use the `cf-xarray` accessor to
        identify coordinates based on CF (Climate and Forecast) conventions
        (e.g., 'latitude', 'longitude').
    2.  **Fallback:** If `cf-xarray` fails, it falls back to a predefined
        list of common, non-standard names (e.g., 'lat', 'lon', 'yc', 'xc', 'y', 'x').

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to inspect.

    Returns
    -------
    tuple[str, str]
        A tuple containing the identified names for the latitude and longitude
        coordinates, respectively.

    Raises
    ------
    ValueError
        If either the latitude or longitude coordinate cannot be identified
        through any of the fallback strategies.
    """
    # Try standard CF names first
    try:
        lat_name = ds.cf["latitude"].name
    except KeyError:
        try:
            lat_name = ds.cf["lat"].name
        except KeyError:
            # Fallback to common non-CF names
            lat_candidates = [
                name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["latitude", "lat", "yc", "y"])
            ]
            if not lat_candidates:
                msg = "Could not identify latitude coordinate"
                raise ValueError(msg) from None
            lat_name = lat_candidates[0]

    try:
        lon_name = ds.cf["longitude"].name
    except KeyError:
        try:
            lon_name = ds.cf["lon"].name
        except KeyError:
            # Fallback to common non-CF names
            lon_candidates = [
                name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["longitude", "lon", "xc", "x"])
            ]
            if not lon_candidates:
                msg = "Could not identify longitude coordinate"
                raise ValueError(msg) from None
            lon_name = lon_candidates[0]

    return str(lat_name), str(lon_name)
