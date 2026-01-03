import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.utils import Grid, create_lat_lon_coords, create_regridding_dataset, format_lat

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from monet_regrid.utils import format_lat
# New import: from monet_regrid.utils import format_lat


def test_format_lat():
    lat_vals = np.arange(-89.5, 89.5 + 1, 1)
    lon_vals = np.arange(-179.5, 179.5 + 1, 1)
    x_vals = np.broadcast_to(lat_vals, (len(lon_vals), len(lat_vals)))
    ds = xr.Dataset(
        data_vars={"x": (("lon", "lat"), x_vals)},
        coords={"lat": lat_vals, "lon": lon_vals},
        attrs={"foo": "bar"},
    )
    ds.lat.attrs["is"] = "coord"
    ds.x.attrs["is"] = "data"

    formatted = format_lat(ds, ds, {"lat": "lat", "lon": "lon"})
    # Check that lat has been extended to include poles
    assert formatted.lat.values[0] == -90
    assert formatted.lat.values[-1] == 90
    # Check that data has been extrapolated to include poles
    assert (formatted.x.isel(lat=0) == -89.5).all()
    assert (formatted.x.isel(lat=-1) == 89.5).all()
    # Check that attrs have been preserved
    assert formatted.attrs["foo"] == "bar"
    assert formatted.lat.attrs["is"] == "coord"
    assert formatted.x.attrs["is"] == "data"


def test_create_lat_lon_coords_returns_dask_arrays():
    """Verify that the coordinate creation function returns lazy Dask arrays."""
    grid = Grid(
        north=90,
        south=-90,
        east=180,
        west=-180,
        resolution_lat=0.1,
        resolution_lon=0.1,
    )
    lat_coords, lon_coords = create_lat_lon_coords(grid)
    assert isinstance(lat_coords, da.Array)
    assert isinstance(lon_coords, da.Array)


def test_create_regridding_dataset_correctness():
    """Test the correctness of the created regridding dataset."""
    grid = Grid(
        north=90,
        south=-90,
        east=180,
        west=-180,
        resolution_lat=0.1,
        resolution_lon=0.1,
    )
    ds = create_regridding_dataset(grid)

    # Verify the shape and content of the coordinates
    assert ds["latitude"].shape == (1801,)
    assert ds["longitude"].shape == (3601,)
    # Use .values to get the computed NumPy array for comparison
    assert ds["latitude"].values.min() == -90
    assert ds["latitude"].values.max() == 90
    assert ds["longitude"].values.min() == -180
    assert ds["longitude"].values.max() == 180
