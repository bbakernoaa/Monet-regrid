"""
Demo of consolidated statistical regridding using the Aero Protocol.
"""

import cartopy.crs as ccrs
import dask.array as da
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def run_demo() -> None:
    """
    Run the regridding demo with high-performance settings.

    Examples
    --------
    >>> run_demo()
    """
    # 1. Create a high-resolution source dataset (Lazy, ~200MB total)
    # Aero Protocol Rule 1: Chunk Awareness (~100MB recommended)
    lat = np.arange(-90, 91, 0.05)  # 3621 points
    lon = np.arange(-180, 181, 0.05)  # 7221 points
    # 3621 * 3610 * 8 bytes = ~104MB per chunk
    data = da.random.random((len(lat), len(lon)), chunks=(len(lat), len(lon) // 2))

    da_source = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="random_data")
    da_source.attrs["units"] = "None"
    da_source.attrs["description"] = "Aero-standard high resolution data"

    # 2. Create a coarse target grid
    target_lat = np.arange(-90, 91, 5)
    target_lon = np.arange(-180, 181, 5)
    ds_target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # 3. Regrid using the consolidated stat method
    regridder = RectilinearRegridder(da_source, ds_target)
    da_regridded = regridder.stat(method="mean")

    # Trigger computation for visualization
    da_regridded = da_regridded.compute()

    # --- TRACK A: Publication Quality (Static) ---
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    da_regridded.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Mean Value"})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title("Aero Consolidated Regrid (Track A: Static)")
    plt.savefig("consolidated_regrid_track_a.png", dpi=300)
    print("Track A plot saved to consolidated_regrid_track_a.png")  # noqa: T201

    # --- TRACK B: Interactive Exploration ---
    # In a notebook, this would be:
    # da_regridded.hvplot(rasterize=True, geo=True, cmap="viridis", title="Aero Track B")
    print("Track B (Interactive): Use .hvplot(rasterize=True) for large grids.")  # noqa: T201


if __name__ == "__main__":
    run_demo()
