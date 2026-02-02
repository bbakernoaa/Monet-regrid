"""
Example of rectilinear regridding visualization following the Aero Protocol.
"""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def create_sample_data() -> xr.Dataset:
    """Create sample rectilinear data for demonstration."""
    lat = np.arange(-90, 91, 2)
    lon = np.arange(0, 360, 2)
    data = np.sin(np.deg2rad(lat[:, None])) * np.cos(np.deg2rad(lon[None, :]))

    ds = xr.Dataset({"temp": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})
    ds.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return ds


def visualize():
    """Demonstrate Track A and Track B visualization."""
    ds = create_sample_data()

    # Define target grid (coarser)
    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(0, 360, 10)
    target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # Import regridder (assuming installed)
    from monet_regrid.core import RectilinearRegridder

    regridder = RectilinearRegridder(ds, target, method="linear")
    regridded = regridder()

    # --- TRACK A: Publication (Matplotlib + Cartopy) ---
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    regridded.temp.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Temperature"})
    ax.set_title("Rectilinear Regridding (Track A - Static)")
    plt.savefig("rectilinear_regrid_static.png")
    print("Saved Track A plot to rectilinear_regrid_static.png")  # noqa: T201

    # --- TRACK B: Exploration (HvPlot) ---
    # Note: This is intended for interactive environments.
    # Here we just show the code structure.
    _plot = regridded.temp.hvplot.quadmesh(
        x="lon", y="lat", geo=True, rasterize=True, cmap="viridis", title="Rectilinear Regridding (Track B - Interactive)"
    )
    # hvplot.save(plot, "rectilinear_regrid_interactive.html") # Requires bokeh/selenium/etc
    print("Track B plot structure created with rasterize=True.")  # noqa: T201


if __name__ == "__main__":
    visualize()
