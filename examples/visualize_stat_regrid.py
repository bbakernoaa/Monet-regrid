"""
Example of statistical regridding visualization following the Aero Protocol.
Demonstrates Track A (Static) and Track B (Interactive).
"""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def create_categorical_data() -> xr.DataArray:
    """Create sample categorical data for demonstration."""
    lat = np.arange(-90, 91, 1)
    lon = np.arange(0, 360, 1)

    # Create some "land cover" types (0: Ocean, 1: Forest, 2: Desert)
    data = np.zeros((len(lat), len(lon)), dtype=int)

    # Simple patterns
    data[100:150, 40:100] = 1
    data[50:80, 200:250] = 2

    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="land_cover")
    da.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    da.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return da


def visualize():
    """Demonstrate Track A and Track B visualization for statistical regridding."""
    da = create_categorical_data()

    # Define a much coarser target grid to show "most common" effect
    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(0, 360, 10)
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # Initialize regridder
    regridder = RectilinearRegridder(None, target)

    # Use "most_common" to regrid categorical data
    regridded = regridder.most_common(values=np.array([0, 1, 2]), data=da)

    # --- TRACK A: Publication (Matplotlib + Cartopy) ---
    fig = plt.figure(figsize=(12, 10))

    # Original
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    da.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap="terrain", cbar_kwargs={"label": "Type"})
    ax1.set_title("Original High-Res Categorical Data")

    # Regridded
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax2.coastlines()
    regridded.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap="terrain", cbar_kwargs={"label": "Type"})
    ax2.set_title("Regridded (Most Common Value) - Track A Static")

    plt.tight_layout()
    plt.savefig("stat_regrid_comparison.png")
    print("Saved Track A plot to stat_regrid_comparison.png")  # noqa: T201

    # --- TRACK B: Exploration (HvPlot) ---
    # Interactive plot with rasterization for performance
    _plot = regridded.hvplot.quadmesh(
        x="lon", y="lat", geo=True, rasterize=True, cmap="terrain", title="Statistical Regridding (Track B - Interactive)"
    )
    print("Track B plot structure created with rasterize=True.")  # noqa: T201


if __name__ == "__main__":
    visualize()
