"""Example of statistical regridding using CurvilinearRegridder."""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def create_example_data():
    """Create a high-resolution curvilinear dataset for demonstration."""
    # Synthetic curvilinear grid (e.g., warped regional grid)
    y, x = np.meshgrid(np.linspace(30, 50, 200), np.linspace(-120, -70, 200))
    # Add some "warping"
    lat_2d = y + 2 * np.sin(x / 5.0)
    lon_2d = x + 2 * np.cos(y / 5.0)

    # Random data with spatial structure
    data = np.exp(-((y - 40) ** 2 + (x + 95) ** 2) / 50.0)
    data += 0.2 * np.random.rand(*y.shape)

    ds = xr.Dataset(
        {"pollution": (("y", "x"), data)},
        coords={
            "lat": (("y", "x"), lat_2d, {"units": "degrees_north"}),
            "lon": (("y", "x"), lon_2d, {"units": "degrees_east"}),
        },
    )
    ds.attrs["history"] = "Generated synthetic curvilinear data."
    return ds


def main():
    # 1. Generate high-res data
    ds_source = create_example_data()

    # 2. Define coarse rectilinear target grid
    ds_target = xr.Dataset(
        coords={
            "latitude": (["latitude"], np.linspace(30, 50, 10), {"units": "degrees_north"}),
            "longitude": (["longitude"], np.linspace(-120, -70, 10), {"units": "degrees_east"}),
        }
    )

    # 3. Initialize data-agnostic regridder
    regridder = CurvilinearRegridder(source_data=None, target_grid=ds_target)

    # 4. Perform statistical reduction (Mean)
    ds_mean = regridder.stat(method="mean", data=ds_source)

    # --- Track A: Static Visualization (Matplotlib + Cartopy) ---
    fig = plt.figure(figsize=(12, 5))

    # Source plot
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ds_source.pollution.plot(
        ax=ax1, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Pollution Level"}
    )
    ax1.coastlines()
    ax1.set_title("Original Curvilinear Grid (High-Res)")

    # Coarsened plot
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ds_mean.plot(
        ax=ax2, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Pollution Level"}
    )
    ax2.coastlines()
    ax2.set_title("Statistical Mean (Coarse Rectilinear)")

    plt.tight_layout()
    plt.savefig("statistical_regrid_example.png")
    print("Static plot saved to statistical_regrid_example.png")  # noqa: T201

    # --- Track B: Interactive Visualization (HvPlot) ---
    # Note: This block is illustrative for interactive environments.
    _interactive_plot = ds_mean.hvplot.quadmesh(
        x="longitude", y="latitude", geo=True, coastline=True, cmap="viridis", title="Mean Pollution (Interactive)", rasterize=True
    )
    # hvplot.save(interactive_plot, "interactive_stat_regrid.html")
    print("Interactive plot logic defined (Track B compliant).")  # noqa: T201


if __name__ == "__main__":
    main()
