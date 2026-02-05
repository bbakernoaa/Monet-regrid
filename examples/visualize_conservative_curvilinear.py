"""
Example script for visualizing conservative curvilinear regridding using the Aero Protocol.
Tracks:
- Track A: Static visualization (Matplotlib + Cartopy)
- Track B: Interactive visualization (HvPlot)
"""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.accessor import Regridder  # noqa: F401


def create_curvilinear_grid_with_bounds(nx=30, ny=30):
    """Create a sample curvilinear dataset with explicit cell bounds.

    Parameters
    ----------
    nx : int, optional
        Number of points in x dimension, by default 30.
    ny : int, optional
        Number of points in y dimension, by default 30.

    Returns
    -------
    xr.Dataset
        A dataset with 2D coordinates and 3D bounds (y, x, vertices).
    """
    lon = np.linspace(-20, 20, nx)
    lat = np.linspace(30, 60, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Add some "curvilinear" distortion (e.g. sheared grid)
    lon2d = lon2d + 2 * np.sin(np.radians(lat2d))
    lat2d = lat2d + 1 * np.cos(np.radians(lon2d))

    # Compute cell boundaries (vertices) for conservative regridding
    # We use a simple approach: midpoints between centers
    def get_bnds(arr):
        # Pad to get edges
        # This is a simplification for demonstration
        bnds = np.zeros((*arr.shape, 4))
        # SW, SE, NE, NW
        # (Very simplified bounding box around center)
        res_x = np.diff(arr, axis=1).mean() if arr.ndim > 1 else 0.5
        res_y = np.diff(arr, axis=0).mean() if arr.ndim > 1 else 0.5

        bnds[:, :, 0] = arr - res_x / 2 - res_y / 2  # SW
        bnds[:, :, 1] = arr + res_x / 2 - res_y / 2  # SE
        bnds[:, :, 2] = arr + res_x / 2 + res_y / 2  # NE
        bnds[:, :, 3] = arr - res_x / 2 + res_y / 2  # NW
        return bnds

    lon_b = get_bnds(lon2d)
    lat_b = get_bnds(lat2d)

    ds = xr.Dataset(
        data_vars={"emissions": (("y", "x"), 100 * np.exp(-((lon2d - 0) ** 2 + (lat2d - 45) ** 2) / 50))},
        coords={
            "lat": (("y", "x"), lat2d, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("y", "x"), lon2d, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("y", "x", "nv"), lat_b, {"units": "degrees_north"}),
            "lon_b": (("y", "x", "nv"), lon_b, {"units": "degrees_east"}),
        },
    ).chunk({"y": 10, "x": 10})  # Use chunks for Lazy by Default (Aero Protocol)
    ds.emissions.attrs["units"] = "kg/m2/s"
    ds.attrs["history"] = "Generated sample curvilinear data with bounds"
    return ds


def create_target_rectilinear_grid(nx=15, ny=15):
    """Create a regular target grid with bounds.

    Parameters
    ----------
    nx : int, optional
        Number of points in x dimension, by default 15.
    ny : int, optional
        Number of points in y dimension, by default 15.

    Returns
    -------
    xr.Dataset
        A dataset with 1D coordinates and 2D bounds.
    """
    lon = np.linspace(-25, 25, nx)
    lat = np.linspace(25, 65, ny)

    # 1D bounds
    def get_1d_bnds(arr):
        res = np.diff(arr).mean()
        bnds = np.stack([arr - res / 2, arr + res / 2], axis=-1)
        return bnds

    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lat, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("lon",), lon, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("lat", "nv"), get_1d_bnds(lat), {"units": "degrees_north"}),
            "lon_b": (("lon", "nv"), get_1d_bnds(lon), {"units": "degrees_east"}),
        }
    )
    return ds


def main():
    # 1. Logic: Perform Conservative Regridding
    print("Preparing data...")  # noqa: T201
    ds_source = create_curvilinear_grid_with_bounds()
    ds_target = create_target_rectilinear_grid()

    print("Running conservative regridding...")  # noqa: T201
    # The CurvilinearRegridder will be used because the source is 2D
    # It supports 'conservative' if bounds are present
    ds_regridded = ds_source.regrid.conservative(ds_target)
    print("Regridding complete.")  # noqa: T201

    # 2. UI: Visualization

    # TRACK A: Static (Publication-ready)
    print("Generating static plot (Track A)...")  # noqa: T201
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot original data on curvilinear grid
    ds_source.emissions.plot(
        ax=axes[0],
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": "Emissions [kg/m2/s]"},
    )
    axes[0].set_title("Source: Curvilinear Grid")
    axes[0].coastlines()
    axes[0].gridlines(draw_labels=True)

    # Plot regridded data on rectilinear grid
    ds_regridded.emissions.plot(
        ax=axes[1],
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": "Emissions [kg/m2/s]"},
    )
    axes[1].set_title("Target: Coarser Rectilinear Grid (Conservative)")
    axes[1].coastlines()
    axes[1].gridlines(draw_labels=True)

    plt.tight_layout()
    output_png = "conservative_regrid_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Static plot saved to {output_png}")  # noqa: T201

    # TRACK B: Interactive (Exploration)
    print("Preparing interactive plot (Track B)...")  # noqa: T201
    # Note: Interactive plot is only fully functional in a Notebook environment
    _interactive_plot = ds_regridded.emissions.hvplot.quadmesh(
        x="lon",
        y="lat",
        rasterize=True,
        geo=True,
        cmap="viridis",
        title="Interactive Conservative Regridded Data",
        coastline=True,
    )
    # In this script, we just demonstrate how it would be created.
    # To view: hvplot.show(interactive_plot)


if __name__ == "__main__":
    main()
