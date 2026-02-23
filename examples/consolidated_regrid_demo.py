"""
Demo of consolidated statistical regridding using the Aero Protocol.
"""
import xarray as xr
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
from monet_regrid.core import RectilinearRegridder

def run_demo():
    # 1. Create a high-resolution source dataset (Lazy)
    lat = np.arange(-90, 91, 0.5)
    lon = np.arange(-180, 181, 0.5)
    data = da.random.random((len(lat), len(lon)), chunks=(181, 361))
    da_source = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="random_data"
    )
    da_source.attrs["units"] = "None"

    # 2. Create a coarse target grid
    target_lat = np.arange(-90, 91, 5)
    target_lon = np.arange(-180, 181, 5)
    ds_target = xr.Dataset(
        coords={"lat": target_lat, "lon": target_lon}
    )

    # 3. Regrid using the consolidated stat method
    regridder = RectilinearRegridder(da_source, ds_target)
    da_regridded = regridder.stat(method="mean")

    # Trigger computation for visualization
    da_regridded = da_regridded.compute()

    # --- TRACK A: Publication Quality (Static) ---
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    da_regridded.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"label": "Mean Value"}
    )
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title("Aero Consolidated Regrid (Track A: Static)")
    plt.savefig("consolidated_regrid_track_a.png", dpi=300)
    print("Track A plot saved to consolidated_regrid_track_a.png")

    # --- TRACK B: Interactive Exploration ---
    # In a notebook, this would be:
    # da_regridded.hvplot(rasterize=True, geo=True, cmap="viridis", title="Aero Track B")
    print("Track B (Interactive): Use .hvplot(rasterize=True) for large grids.")

if __name__ == "__main__":
    run_demo()
