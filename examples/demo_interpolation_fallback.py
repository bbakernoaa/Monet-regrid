import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def visualize_interpolation_fallback():
    """Demonstrate the InterpolationEngine with a visual comparison."""
    # 1. Setup Source Grid (Curvilinear)
    lon, lat = np.meshgrid(np.linspace(-20, 20, 40), np.linspace(40, 60, 40))
    # Add some noise to make it curvilinear
    lon += np.random.uniform(-0.1, 0.1, lon.shape)
    lat += np.random.uniform(-0.1, 0.1, lat.shape)

    data = np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    source_ds = xr.Dataset({"sample": (("y", "x"), data)}, coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)})
    source_ds.lon.attrs["units"] = "degrees_east"
    source_ds.lat.attrs["units"] = "degrees_north"

    # 2. Setup Target Grid (Rectilinear)
    target_ds = xr.Dataset(coords={"lat": (("lat",), np.linspace(40, 60, 100)), "lon": (("lon",), np.linspace(-20, 20, 100))})

    # 3. Regrid using Fallback (Disable Numba)
    from monet_regrid.interpolation import core

    original_has_numba = core.HAS_NUMBA
    core.HAS_NUMBA = False

    try:
        regridder = CurvilinearRegridder(source_ds, target_ds, method="linear")
        regridded = regridder()

        # 4. Visualization (Track A: Publication)
        fig = plt.figure(figsize=(12, 5))

        # Source
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_title("Source (Curvilinear)")
        p1 = ax1.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap="viridis")
        ax1.coastlines()
        plt.colorbar(p1, ax=ax1, shrink=0.5)

        # Regridded (Fallback Path)
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.set_title("Regridded (Linear Fallback)")
        p2 = ax2.pcolormesh(target_ds.lon, target_ds.lat, regridded.sample, transform=ccrs.PlateCarree(), cmap="viridis")
        ax2.coastlines()
        plt.colorbar(p2, ax=ax2, shrink=0.5)

        plt.tight_layout()
        plt.savefig("interpolation_fallback_demo.png")
        # print is used for user feedback in this script
        print("Visualization saved to interpolation_fallback_demo.png")  # noqa: T201

    finally:
        core.HAS_NUMBA = original_has_numba


if __name__ == "__main__":
    visualize_interpolation_fallback()
