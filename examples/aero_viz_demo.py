"""
🍃⚡ Aero Visualization Demo: Track A (Publication) & Track B (Exploration)

This example demonstrates the Aero Protocol's visualization requirements for
Earth science data. It shows how to create publication-quality static maps
using Cartopy and interactive dashboards using HvPlot.

Protocol Rules:
1. Track A (Static): Must include `projection=` in axes and `transform=` in plot calls.
2. Track B (Interactive): Must use `rasterize=True` for large grids.
"""

import cartopy.crs as ccrs
import dask.array as da
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import xarray as xr

from monet_regrid.utils import Grid, create_regridding_dataset


def create_demo_data() -> xr.Dataset:
    """Create a global lazy dataset for demonstration.

    Returns
    -------
    xr.Dataset
        A lazy xarray dataset containing synthetic spatial data.

    Examples
    --------
    >>> ds = create_demo_data()
    >>> print(ds.synthetic_data.chunks)
    ((181,), (361,))
    """
    grid = Grid(north=90, south=-90, east=180, west=-180, resolution_lat=1, resolution_lon=1)
    ds = create_regridding_dataset(grid, chunks=180)

    # Add a data variable with some pattern
    lat, lon = ds.latitude, ds.longitude
    # Broadcast to 2D for computation
    lat_2d, lon_2d = xr.broadcast(lat, lon)
    # Ensure we use the underlying dask arrays for computation
    data = da.sin(da.deg2rad(lat_2d.data)) * da.cos(da.deg2rad(lon_2d.data))
    ds["synthetic_data"] = (("latitude", "longitude"), data)
    ds["synthetic_data"].attrs["units"] = "Aero-Units"
    ds.attrs["history"] = "Created demo data for Aero visualization"

    return ds


def track_a_publication_map(ds: xr.Dataset) -> None:
    """Track A: Static Publication-Quality Map (Matplotlib + Cartopy).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to visualize.

    Returns
    -------
    None
        Saves the plot to 'aero_track_a_demo.png'.

    Examples
    --------
    >>> ds = create_demo_data()
    >>> track_a_publication_map(ds)
    """
    print("Generating Track A: Publication-quality map...")  # noqa: T201

    plt.figure(figsize=(12, 6))
    # Rule: Mandatory projection in axes
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Rule: Mandatory transform in plot calls
    ds.synthetic_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"label": "Aero Intensity"},
    )

    plt.title("Aero Protocol Track A: Static Map (Cartopy)")
    plt.savefig("aero_track_a_demo.png", dpi=300, bbox_inches="tight")
    print("Track A saved to aero_track_a_demo.png")  # noqa: T201


def track_b_interactive_map(ds: xr.Dataset) -> None:
    """Track B: Interactive Exploration Map (HvPlot / Holoviews).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to visualize.

    Returns
    -------
    None
        Saves the plot to 'aero_track_b_demo.html'.

    Examples
    --------
    >>> ds = create_demo_data()
    >>> track_b_interactive_map(ds)
    """
    print("Generating Track B: Interactive exploration map...")  # noqa: T201

    # Rule: Mandatory rasterize=True for large grids to maintain performance
    plot = ds.synthetic_data.hvplot(
        x="longitude",
        y="latitude",
        geo=True,
        coastline=True,
        projection=ccrs.PlateCarree(),
        rasterize=True,
        cmap="viridis",
        title="Aero Protocol Track B: Interactive Map (HvPlot)",
    )

    # Save as HTML for demonstration
    import holoviews as hv

    hv.save(plot, "aero_track_b_demo.html")
    print("Track B saved to aero_track_b_demo.html")  # noqa: T201


if __name__ == "__main__":
    # Create lazy dataset
    ds = create_demo_data()

    # Run visualization tracks
    track_a_publication_map(ds)

    try:
        track_b_interactive_map(ds)
    except Exception as e:
        print(f"Track B failed (likely missing backend components): {e}")  # noqa: T201

    print("\n🍃⚡ Aero Visualization Demo Complete.")  # noqa: T201
