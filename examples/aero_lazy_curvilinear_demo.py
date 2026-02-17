"""
Aero Protocol Demo: Lazy Curvilinear Regridding
===============================================

This script demonstrates the Aero Protocol's standards for lazy evaluation,
scientific hygiene, and the two-track visualization workflow.

Standards:
1. Speed: Lazy initialization and Dask-backed computation.
2. Maintainability: Strict type hints and NumPy-style docstrings.
3. Provenance: Automatically updated history attributes.
4. Visualization: Track A (Publication/Static) and Track B (Interactive).
"""

import cartopy.crs as ccrs
import dask.array as da
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.accessor import Regridder  # noqa: F401


def create_synthetic_curvilinear_data(ny: int = 100, nx: int = 200) -> xr.Dataset:
    """Create a synthetic lazy curvilinear dataset.

    Parameters
    ----------
    ny : int, optional
        Number of points in y-dimension, by default 100.
    nx : int, optional
        Number of points in x-dimension, by default 200.

    Returns
    -------
    xr.Dataset
        A Dask-backed dataset with 2D curvilinear coordinates.
    """
    # Create 2D coordinates
    lons_1d = np.linspace(-180, 180, nx)
    lats_1d = np.linspace(-90, 90, ny)
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)

    # Add some "curviness"
    lat2d = lat2d + np.sin(lon2d * np.pi / 180.0) * 5

    # Wrap in Dask arrays
    lat_da = da.from_array(lat2d, chunks=(ny // 2, nx // 2))
    lon_da = da.from_array(lon2d, chunks=(ny // 2, nx // 2))

    # Create data
    data = da.random.random((ny, nx), chunks=(ny // 2, nx // 2))

    ds = xr.Dataset(
        data_vars={"sample_data": (("y", "x"), data)},
        coords={
            "latitude": (("y", "x"), lat_da, {"units": "degrees_north"}),
            "longitude": (("y", "x"), lon_da, {"units": "degrees_east"}),
        },
        attrs={"history": "Created synthetic curvilinear data"},
    )
    return ds


def run_demo() -> None:
    """Execute the Aero lazy curvilinear regridding demo."""
    print("--- Aero Protocol: Lazy Curvilinear Demo ---")  # noqa: T201

    # 1. Create source data (Lazy)
    print("Step 1: Creating synthetic curvilinear data...")  # noqa: T201
    ds_source = create_synthetic_curvilinear_data(400, 800)
    print(f"Source data is lazy: {isinstance(ds_source.sample_data.data, da.Array)}")  # noqa: T201

    # 2. Create target grid (Lazy)
    print("Step 2: Creating target rectilinear grid...")  # noqa: T201
    ds_target = xr.Dataset(
        coords={
            "lat": (("lat",), np.arange(-80, 81, 2), {"units": "degrees_north"}),
            "lon": (("lon",), np.arange(-170, 171, 2), {"units": "degrees_east"}),
        }
    )

    # 3. Initialize Regridder (Instant/Lazy due to recent refactor)
    print("Step 3: Initializing regridder...")  # noqa: T201
    regridder = ds_source.regrid.build_regridder(ds_target, method="nearest")
    print("Regridder initialized successfully without eager computation.")  # noqa: T201

    # 4. Perform Regridding (Lazy)
    print("Step 4: Executing regridding...")  # noqa: T201
    ds_regridded = regridder()
    print(f"Regridded data is lazy: {isinstance(ds_regridded.sample_data.data, da.Array)}")  # noqa: T201
    print(f"Provenance tracking:\n{ds_regridded.attrs.get('history', 'N/A')}")  # noqa: T201

    # 5. Visualization (Two-Track Rule)
    print("\nStep 5: Visualization (Two-Track Rule)")  # noqa: T201

    # Track A: Publication-Quality (Matplotlib + Cartopy)
    print("- Track A: Generating static plot (matplotlib/cartopy)...")  # noqa: T201
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    # Compute a small sample for plotting
    plot_data = ds_regridded.sample_data.compute()

    plot_data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"label": "Random Values"})
    ax.set_title("Aero Track A: Static Curvilinear-to-Rectilinear Regridding")
    plt.savefig("aero_demo_track_a.png")
    print("  Saved Track A to aero_demo_track_a.png")  # noqa: T201

    # Track B: Interactive Exploration (HvPlot)
    print("- Track B: Interactive code (Ready for Jupyter)...")  # noqa: T201
    # Note: This returns an object that displays interactively in Jupyter
    interactive_plot = ds_regridded.sample_data.hvplot.image(
        x="lon",
        y="lat",
        rasterize=True,  # Mandatory for large grids
        cmap="viridis",
        title="Aero Track B: Interactive Exploration",
    )
    # Since we are in a script, we just acknowledge the object creation
    print(f"  Interactive plot object created: {type(interactive_plot)}")  # noqa: T201
    plt.close(fig)


if __name__ == "__main__":
    run_demo()
