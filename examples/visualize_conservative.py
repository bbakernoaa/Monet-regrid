"""Visualization script for conservative regridding demonstration."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import monet_regrid  # noqa: F401


def run_visualization() -> None:
    """Generate and save a demonstration plot for conservative regridding."""
    # 1. Create a global source grid with some features
    lats = np.arange(-90, 91, 1)
    lons = np.arange(0, 360, 1)
    lon2d, lat2d = np.meshgrid(lons, lats)
    data = np.sin(np.deg2rad(lat2d)) * np.cos(np.deg2rad(lon2d))

    ds_source = xr.Dataset(
        {"field": (("lat", "lon"), data)},
        coords={"lat": (["lat"], lats, {"units": "degrees_north"}), "lon": (["lon"], lons, {"units": "degrees_east"})},
    )

    # 2. Define a coarser target grid
    ds_target = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-80, 81, 10), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0, 360, 20), {"units": "degrees_east"}),
        }
    )

    # 3. Regrid
    ds_regrid = ds_source.regrid.conservative(ds_target)

    # 4. Visualization (Track A: Publication Quality)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Plot regridded data
    # Note: ccrs.PlateCarree() is the transform for the input data coords
    ds_regrid.field.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", cbar_kwargs={"label": "Conservative Regridded Field"})

    ax.set_title("Aero Protocol Demo: Conservative Regridding (Global)")
    plt.savefig("conservative_regrid_demo.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    run_visualization()
