"""
🍃⚡ Aero Protocol Visualization: Curvilinear Longitude Alignment Fix

This example demonstrates the fix for curvilinear statistical regridding
when longitude ranges differ between source and target grids.

Track A: Publication Quality (Matplotlib + Cartopy)
Track B: Interactive Exploration (HvPlot)
"""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def run_demo():
    # 1. Create source curvilinear grid [180, 360]
    # Representing a grid that starts at the antimeridian
    lon_1d = np.linspace(180, 360, 50)
    lat_1d = np.linspace(-60, 60, 30)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Add some noise to make it "truly" curvilinear
    lon_2d += np.random.uniform(-0.1, 0.1, lon_2d.shape)
    lat_2d += np.random.uniform(-0.1, 0.1, lat_2d.shape)

    data = np.sin(np.radians(lon_2d)) * np.cos(np.radians(lat_2d))

    source_da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "longitude": (("y", "x"), lon_2d),
            "latitude": (("y", "x"), lat_2d),
        },
        name="source_data",
    )

    # 2. Create target rectilinear grid [-180, 180]
    target_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.linspace(-60, 60, 20)),
            "lon": (["lon"], np.linspace(-180, 180, 40)),
        }
    )

    # 3. Regrid using stat (Mean)
    # The fix ensures that [180, 360] is shifted to [-180, 0] to match the target
    regridder = CurvilinearRegridder(source_da, target_ds)
    result = regridder.stat(method="mean")

    # --- TRACK A: Publication (Static) ---
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot regridded data
    # Mandatory: transform=ccrs.PlateCarree()
    result.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Regridded Value"})

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title("🍃⚡ Curvilinear to Rectilinear (Stat Mean) - Longitude Alignment Fix")

    plt.tight_layout()
    plt.savefig("curvilinear_fix_track_a.png")

    # --- TRACK B: Exploration (Interactive) ---
    # In a real environment, this would display an interactive plot
    # Mandatory: rasterize=True for large grids
    result.hvplot.quadmesh(
        x="lon",
        y="lat",
        projection=ccrs.PlateCarree(),
        rasterize=True,
        cmap="viridis",
        title="Interactive Exploration: Curvilinear Fix",
    )
    # hvplot.save(interactive_plot, 'curvilinear_fix_track_b.html')


if __name__ == "__main__":
    run_demo()
