"""
Example script for visualizing curvilinear regridding using the Aero Protocol.
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


def create_sample_data():
    """Create a sample curvilinear dataset."""
    lon = np.linspace(-180, 180, 50)
    lat = np.linspace(-90, 90, 25)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Add some "curvilinear" distortion
    lon2d = lon2d + 5 * np.sin(np.radians(lat2d))

    ds = xr.Dataset(
        data_vars={"temperature": (("y", "x"), 15 + 10 * np.cos(np.radians(lat2d)))},
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
    )
    ds.temperature.attrs["units"] = "degC"
    ds.attrs["history"] = "Generated sample data"
    return ds


def create_target_grid():
    """Create a regular target grid."""
    lon = np.linspace(-180, 180, 100)
    lat = np.linspace(-90, 90, 50)
    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lat, {"units": "degrees_north"}),
            "lon": (("lon",), lon, {"units": "degrees_east"}),
        }
    )
    return ds


def main():
    # 1. Logic: Perform regridding
    ds_source = create_sample_data()
    ds_target = create_target_grid()

    ds_regridded = ds_source.regrid.linear(ds_target)

    # 2. UI: Visualization

    # TRACK A: Static (Publication-ready)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={"projection": ccrs.PlateCarree()})

    ds_source.temperature.plot(ax=axes[0], x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="RdYlBu_r", add_colorbar=True)
    axes[0].set_title("Original Curvilinear Grid")
    axes[0].coastlines()

    ds_regridded.temperature.plot(ax=axes[1], x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="RdYlBu_r", add_colorbar=True)
    axes[1].set_title("Regridded Rectilinear Grid")
    axes[1].coastlines()

    plt.savefig("regridding_comparison.png", dpi=300, bbox_inches="tight")

    # TRACK B: Interactive (Exploration)
    # Note: In a headless environment, this just prepares the object.
    _interactive_plot = ds_regridded.temperature.hvplot.quadmesh(
        x="lon", y="lat", rasterize=True, geo=True, cmap="RdYlBu_r", title="Interactive Regridded Data"
    )
    # To view this, one would usually use: hvplot.show(_interactive_plot)


if __name__ == "__main__":
    main()
