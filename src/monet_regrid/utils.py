from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypedDict, overload

import cf_xarray  # noqa: F401
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


class InvalidBoundsError(Exception): ...


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

        Args:
            grid: Grid object containing the bounds and resolution of the
                cartesian grid.
            lat_name: Name for the latitudinal coordinate and dimension.
                Defaults to "latitude".
            lon_name: Name for the longitudinal coordinate and dimension.
                Defaults to "longitude".

        Returns:
            A dataset with the latitude and longitude coordinates corresponding to the
                specified grid. Contains no data variables.
        """
        return create_regridding_dataset(self, lat_name, lon_name)


def create_lat_lon_coords(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Create latitude and longitude coordinates based on the provided grid parameters.

    Args:
        grid: Grid object.

    Returns:
        Latititude coordinates, longitude coordinates.
    """

    if np.remainder((grid.north - grid.south), grid.resolution_lat) > 0:
        lat_coords = np.arange(grid.south, grid.north, grid.resolution_lat)
    else:
        lat_coords = np.arange(grid.south, grid.north + grid.resolution_lat, grid.resolution_lat)

    if np.remainder((grid.east - grid.west), grid.resolution_lat) > 0:
        lon_coords = np.arange(grid.west, grid.east, grid.resolution_lon)
    else:
        lon_coords = np.arange(grid.west, grid.east + grid.resolution_lon, grid.resolution_lon)
    return lat_coords, lon_coords


def create_regridding_dataset(grid: Grid, lat_name: str = "latitude", lon_name: str = "longitude") -> xr.Dataset:
    """Create a dataset to use for regridding.

    Args:
        grid: Grid object containing the bounds and resolution of the cartesian grid.
        lat_name: Name for the latitudinal coordinate and dimension.
            Defaults to "latitude".
        lon_name: Name for the longitudinal coordinate and dimension.
            Defaults to "longitude".

    Returns:
        A dataset with the latitude and longitude coordinates corresponding to the
            specified grid. Contains no data variables.
    """
    lat_coords, lon_coords = create_lat_lon_coords(grid)
    return xr.Dataset(
        {
            lat_name: ([lat_name], lat_coords, {"units": "degrees_north"}),
            lon_name: ([lon_name], lon_coords, {"units": "degrees_east"}),
        }
    )


def to_intervalindex(coords: np.ndarray) -> pd.IntervalIndex:
    """Convert a 1-d coordinate array to a pandas IntervalIndex. Take
    the midpoints between the coordinates as the interval boundaries.

    Args:
        coords: 1-d array containing the coordinate values.

    Returns:
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

    Args:
        a: Pandas IntervalIndex containing the first set of intervals.
        b: Pandas IntervalIndex containing the second set of intervals.

    Returns:
        2D numpy array containing overlap (as a fraction) between the intervals of a
            and b. If there is no overlap, the value will be 0.
    """
    # TODO: newaxis on B and transpose is MUCH faster on benchmark.
    #  likely due to it being the bigger dimension.
    #  size(a) > size(b) leads to better perf than size(b) > size(a)
    mins = np.minimum(a.right.to_numpy(), b.right.to_numpy()[:, np.newaxis])
    maxs = np.maximum(a.left.to_numpy(), b.left.to_numpy()[:, np.newaxis])
    overlap: np.ndarray = np.maximum(mins - maxs, 0).T
    return overlap


def normalize_overlap(overlap: np.ndarray) -> np.ndarray:
    """Normalize overlap values so they sum up to 1.0 along the first axis."""
    overlap_sum: np.ndarray = overlap.sum(axis=0)
    overlap_sum[overlap_sum == 0] = 1e-12  # Avoid dividing by 0.
    return overlap / overlap_sum


def create_dot_dataarray(
    weights: np.ndarray,
    coord: str,
    target_coords: np.ndarray,
    source_coords: np.ndarray,
) -> xr.DataArray:
    """Create a DataArray to be used at dot product compatible with xr.dot."""
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
    """Return a set of coords which two dataset/arrays have in common."""
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
    """Use to call a function that expects a Dataset on either a Dataset or
    DataArray, round-tripping to a temporary dataset."""
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
) -> xr.Dataset: ...


@overload
def format_for_regrid(
    obj: xr.DataArray,
    target: xr.Dataset,
    stats: bool = False,
) -> xr.DataArray: ...


def format_for_regrid(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    stats: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Apply any pre-formatting to the input dataset to prepare for regridding.
    Currently handles padding of spherical geometry if lat/lon coordinates can
    be inferred and the domain size requires boundary padding.
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

    return result


def format_lat(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,  # noqa ARG001
    formatted_coords: dict[str, str],
) -> xr.DataArray | xr.Dataset:
    """If the latitude coordinate is inferred to be global, defined as having
    a value within one grid spacing of the poles, and the grid does not natively
    have values at -90 and 90, add a single value at each pole computed as the
    mean of the first and last latitude bands. This should be roughly equivalent
    to the `Pole="all"` option in `ESMF`.

    For example, with a grid spacing of 1 degree, and a source grid ranging from
    -89.5 to 89.5, the poles would be padded with values at -90 and 90. A grid ranging
    from -88 to 88 would not be padded because coverage does not extend all the way
    to the poles. A grid ranging from -90 to 90 would also not be padded because the
    poles will already be covered in the regridding weights.

    Note: Pole padding is only applied to 1D latitude coordinates (rectilinear grids).
    For 2D coordinates (curvilinear grids), no padding is performed since the grid
    structure is irregular and pole padding doesn't apply.
    """
    lat_coord = formatted_coords["lat"]
    lon_coord = formatted_coords.get("lon")

    # Check if this is a 2D coordinate (curvilinear grid)
    lat_vals = obj.coords[lat_coord].values
    if lat_vals.ndim == 2:
        # For curvilinear grids, skip pole padding
        return obj

    # Concat a padded value representing the mean of the first/last lat bands
    # This should match the Pole="all" option of ESMF
    # TODO: with cos(90) = 0 weighting, these weights might be 0?

    polar_lat = 90
    dy: Any = obj.coords[lat_coord].diff(lat_coord).max().values.item()

    # Only pad if global but don't have edge values directly at poles
    # NOTE: could use xr.pad here instead of xr.concat, but none of the
    # modes are an exact fit for this scheme

    lat_dim = obj[lat_coord].dims[0]
    lon_dim = obj[lon_coord].dims[0] if lon_coord else None

    # South pole
    if dy - polar_lat >= obj.coords[lat_coord].values[0] > -polar_lat:
        south_pole = obj.isel({lat_dim: 0})
        if lon_dim is not None:
            south_pole = south_pole.mean(lon_dim, keep_attrs=True)
        obj = xr.concat([south_pole, obj], dim=lat_dim)  # type: ignore
        lat_vals = np.concatenate([[-polar_lat], lat_vals])

    # North pole
    if polar_lat - dy <= obj.coords[lat_coord].values[-1] < polar_lat:
        north_pole = obj.isel({lat_dim: -1})
        if lon_dim is not None:
            north_pole = north_pole.mean(lon_dim, keep_attrs=True)
        obj = xr.concat([obj, north_pole], dim=lat_dim)  # type: ignore
        lat_vals = np.concatenate([lat_vals, [polar_lat]])

    obj = update_coord(obj, lat_coord, lat_vals)

    return obj


def format_lon(
    obj: xr.DataArray | xr.Dataset, target: xr.Dataset, formatted_coords: dict[str, str]
) -> xr.DataArray | xr.Dataset:
    """Format the longitude coordinate by shifting the source grid to line up with
    the target anywhere in the range of -360 to 360, and then add a single wraparound
    padding column if the domain is inferred to be global and the east or west edges
    of the target lie outside the source grid centers.

    For example, with a source grid ranging from 0.5 to 359.5 and a target grid ranging
    from -180 to 180, the source grid would be shifted to -179.5 to 179.5 and then
    padded on both the left and right with wraparound values at -180.5 and 180.5 to
    provide full coverage for the target edge cells at -180 and 180.

    Note: Longitude formatting is only applied to 1D longitude coordinates (rectilinear grids).
    For 2D coordinates (curvilinear grids), no formatting is performed since the grid
    structure is irregular and wraparound logic doesn't apply.
    """
    lon_coord = formatted_coords["lon"]

    # Check if this is a 2D coordinate (curvilinear grid)
    lon_vals = obj.coords[lon_coord].values
    if lon_vals.ndim == 2:
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
    source_vals = obj.coords[lon_coord].values
    target_vals = target.coords[target_lon_coord].values
    wrap_point = (target_vals[-1] + target_vals[0] + 360) / 2
    source_vals = np.where(source_vals < wrap_point - 360, source_vals + 360, source_vals)
    source_vals = np.where(source_vals > wrap_point, source_vals - 360, source_vals)
    obj = update_coord(obj, lon_coord, source_vals)

    obj = ensure_monotonic(obj, lon_coord)

    # Only pad if domain is global in lon
    source_lon = obj.coords[lon_coord]
    target_lon = target.coords[target_lon_coord]
    dx_s: Any = source_lon.diff(lon_coord).max().values.item()
    dx_t: Any = target_lon.diff(target_lon_coord).max().values.item()
    is_global_lon = source_lon.max().values - source_lon.min().values >= 360 - dx_s

    if is_global_lon:
        left_pad = (source_lon.values[0] - target_lon.values[0] + dx_t / 2) / dx_s
        right_pad = (target_lon.values[-1] - source_lon.values[-1] + dx_t / 2) / dx_s
        left_pad = int(np.ceil(np.max([left_pad, 0])))
        right_pad = int(np.ceil(np.max([right_pad, 0])))
        obj = obj.pad({lon_coord: (left_pad, right_pad)}, mode="wrap", keep_attrs=True)
        lon_vals = obj.coords[lon_coord].values
        if left_pad:
            lon_vals[:left_pad] = source_lon.values[-left_pad:] - 360
        if right_pad:
            lon_vals[-right_pad:] = source_lon.values[:right_pad] + 360
        obj = update_coord(obj, lon_coord, lon_vals)
        obj = ensure_monotonic(obj, lon_coord)

    return obj


def coord_is_covered(obj: xr.DataArray | xr.Dataset, target: xr.Dataset, coord: Hashable) -> bool:
    """Check if the source coord fully covers the target coord."""
    pad = target[coord].diff(coord).max().values
    left_covered = obj[coord].min() <= target[coord].min() - pad
    right_covered = obj[coord].max() >= target[coord].max() + pad
    return bool(left_covered.item() and right_covered.item())


@overload
def ensure_monotonic(obj: xr.DataArray, coord: Hashable) -> xr.DataArray: ...


@overload
def ensure_monotonic(obj: xr.Dataset, coord: Hashable) -> xr.Dataset: ...


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
def update_coord(obj: xr.DataArray, coord: Hashable, coord_vals: np.ndarray) -> xr.DataArray: ...


@overload
def update_coord(obj: xr.Dataset, coord: Hashable, coord_vals: np.ndarray) -> xr.Dataset: ...


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

    Args:
        ds: Input xarray dataset

    Returns:
        GridType: Either GridType.RECTILINEAR or GridType.CURVILINEAR

    Raises:
        ValueError: If coordinates cannot be identified or if mixed dimensions are found
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
            lat_coord_names = [
                name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["lat", "yc", "y"])
            ]
            lon_coord_names = [
                name for name in ds.coords if any(keyword in str(name).lower() for keyword in ["lon", "xc", "x"])
            ]

            # If we have both lat and lon coordinates, check their dimensions
            if lat_coord_names and lon_coord_names:
                lat_coord = ds[lat_coord_names[0]]
                lon_coord = ds[lon_coord_names[0]]

                # Check the number of dimensions for each coordinate
                lat_ndim = lat_coord.ndim
                lon_ndim = lon_coord.ndim

                # Both coordinates should have the same number of dimensions
                if lat_ndim != lon_ndim:
                    msg = (
                        f"Mismatched coordinate dimensions: latitude has {lat_ndim} dims, longitude has {lon_ndim} dims"
                    )
                    raise ValueError(msg) from None

                # Determine grid type based on dimensionality
                if lat_ndim == 1:
                    return GridType.RECTILINEAR
                elif lat_ndim == 2:
                    # Check if 2D coordinates are actually just meshgrid of 1D coordinates
                    # In such cases, we should still treat them as rectilinear
                    try:
                        # For true rectilinear grids with 2D coordinates,
                        # lat varies only along one dimension and lon varies only along another
                        # Check if latitude is constant along the second axis (x-direction)
                        np.any(np.diff(lat_coord.values, axis=1) != 0)
                        # Check if longitude is constant along the first axis (y-direction)
                        np.any(np.diff(lon_coord.values, axis=0) != 0)

                        # If lat only varies in y and lon only varies in x, it's still
                        # treated as curvilinear since rectilinear interpolation expects 1D coordinates
                        return GridType.CURVILINEAR
                    except (IndexError, AttributeError):
                        # If we can't determine, default to curvilinear for safety
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
                        if (
                            "lat" in str(coord_name).lower()
                            or "yc" in str(coord_name).lower()
                            or "y" in str(coord_name).lower()
                        ):
                            potential_lat_coords.append(coord_name)
                        elif (
                            "lon" in str(coord_name).lower()
                            or "xc" in str(coord_name).lower()
                            or "x" in str(coord_name).lower()
                        ):
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

    Args:
        source_ds: Source dataset
        target_ds: Target dataset

    Returns:
        Tuple of (source_grid_type, target_grid_type)

    Raises:
        ValueError: If either grid has an unsupported type
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
) -> xr.Dataset: ...


@overload
def validate_input(
    data: xr.DataArray,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,
) -> xr.Dataset: ...


def validate_input(
    data: xr.DataArray | xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,  # noqa: ARG001
) -> xr.Dataset:
    # Check for coordinate compatibility using semantic matching instead of exact name matching
    # This allows latitude/longitude to match with lat/lon, etc.

    def _find_coordinate_matches(source_coords: list[Hashable], target_coords: list[Hashable]) -> list[Hashable]:
        """Find semantic matches between coordinate names."""
        matches: list[Hashable] = []

        # Define coordinate name patterns
        lat_patterns = ["lat", "latitude", "y", "yc"]
        lon_patterns = ["lon", "longitude", "x", "xc"]

        source_lat_coords = [c for c in source_coords if any(p in str(c).lower() for p in lat_patterns)]
        source_lon_coords = [c for c in source_coords if any(p in str(c).lower() for p in lon_patterns)]
        target_lat_coords = [c for c in target_coords if any(p in str(c).lower() for p in lat_patterns)]
        target_lon_coords = [c for c in target_coords if any(p in str(c).lower() for p in lon_patterns)]

        # If we have both lat and lon coordinates in both source and target, we have matches
        if source_lat_coords and source_lon_coords and target_lat_coords and target_lon_coords:
            matches.extend(source_lat_coords[:1])  # Take first match
            matches.extend(source_lon_coords[:1])  # Take first match

        # Also check for exact coordinate name matches
        exact_matches = set(source_coords).intersection(set(target_coords))
        matches.extend(exact_matches)

        return matches

    # Check coordinate compatibility
    coord_matches = _find_coordinate_matches(list(data.coords), list(ds_target_grid.coords))

    if len(coord_matches) == 0:
        # Only check dimensions if no coordinate matches found
        dim_matches = set(data.dims).intersection(set(ds_target_grid.dims))

        if len(dim_matches) == 0:
            # As a last resort, check for semantic dimension matches
            semantic_dim_matches = _find_coordinate_matches(list(data.dims), list(ds_target_grid.dims))

            if len(semantic_dim_matches) == 0:
                msg = (
                    "No compatible coordinates or dimensions found between source and target:\n"
                    " regridding is not possible.\n"
                    f"Target coords: {list(ds_target_grid.coords)}\n"
                    f"Source coords: {list(data.coords)}\n"
                    f"Target dims: {list(ds_target_grid.dims)}\n"
                    f"Source dims: {list(data.dims)}"
                )
                raise ValueError(msg)

    return ds_target_grid


def _create_cache_key(data: xr.DataArray | xr.Dataset, time_dim: str | None = None) -> tuple:
    """
    Create a stable cache key from an xarray object's metadata.

    This key is based on the coordinates and dimensions, making it suitable for
    caching operations that depend on the grid structure rather than the data values.

    Args:
        data: The xarray DataArray or Dataset.
        time_dim: The name of the time dimension, if any.

    Returns:
        A hashable tuple that serves as a cache key.
    """
    # Create a hashable representation of the coordinates
    # Includes name, shape, dtype, and the raw values as bytes
    coords_key = frozenset(
        (name, coord.shape, coord.dtype, coord.values.tobytes()) for name, coord in data.coords.items()
    )

    # The dimensions are also important
    dims_key = tuple(sorted(data.dims))

    return (coords_key, dims_key, time_dim)


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
                name
                for name in ds.coords
                if any(keyword in str(name).lower() for keyword in ["latitude", "lat", "yc", "y"])
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
                name
                for name in ds.coords
                if any(keyword in str(name).lower() for keyword in ["longitude", "lon", "xc", "x"])
            ]
            if not lon_candidates:
                msg = "Could not identify longitude coordinate"
                raise ValueError(msg) from None
            lon_name = lon_candidates[0]

    return str(lat_name), str(lon_name)
