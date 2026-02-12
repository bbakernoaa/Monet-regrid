"""
Comparison of multiple curvilinear regridding methods.

This script demonstrates:
- Linear and Nearest neighbor interpolation
- Conservative regridding (Area-weighted)
- Bilinear and Cubic spline interpolation
- Visualization Track A (Publication-ready with Cartopy)
- Visualization Track B (Interactive with HvPlot)

Follows the Aero Protocol for visualization and performance.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_regrid.accessor import Regridder  # noqa: F401


def create_source_grid(nx: int = 40, ny: int = 40) -> xr.Dataset:
    """Create a sample curvilinear dataset with explicit cell bounds.

    Parameters
    ----------
    nx : int, optional
        Number of grid points in x, by default 40.
    ny : int, optional
        Number of grid points in y, by default 40.

    Returns
    -------
    xr.Dataset
        A dataset with 2D coordinates and 3D bounds, chunked for lazy evaluation.

    Examples
    --------
    >>> ds = create_source_grid(10, 10)
    >>> ds.chunks is not None
    True
    """
    lon = np.linspace(-30, 30, nx)
    lat = np.linspace(20, 60, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Apply distortion
    lon2d = lon2d + 5 * np.sin(np.radians(lat2d))
    lat2d = lat2d + 3 * np.cos(np.radians(lon2d))

    # Signal: A peak in the center
    data = 100 * np.exp(-((lon2d - 0) ** 2 + (lat2d - 40) ** 2) / 100)

    # Bounds for conservative
    def get_bnds(arr: np.ndarray) -> np.ndarray:
        """Compute bounding vertices around center coordinates.

        Parameters
        ----------
        arr : np.ndarray
            Center coordinates (ny, nx).

        Returns
        -------
        np.ndarray
            Vertices (ny, nx, 4).
        """
        bnds = np.zeros((*arr.shape, 4))
        res_x = np.diff(arr, axis=1).mean()
        res_y = np.diff(arr, axis=0).mean()
        bnds[:, :, 0] = arr - res_x / 2 - res_y / 2
        bnds[:, :, 1] = arr + res_x / 2 - res_y / 2
        bnds[:, :, 2] = arr + res_x / 2 + res_y / 2
        bnds[:, :, 3] = arr - res_x / 2 + res_y / 2
        return bnds

    ds = xr.Dataset(
        data_vars={"var": (("y", "x"), data)},
        coords={
            "lat": (("y", "x"), lat2d, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("y", "x"), lon2d, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("y", "x", "nv"), get_bnds(lat2d)),
            "lon_b": (("y", "x", "nv"), get_bnds(lon2d)),
        },
    ).chunk({"y": 20, "x": 20})
    return ds


def create_target_grid(nx: int = 20, ny: int = 20) -> xr.Dataset:
    """Create a regular target grid with bounds.

    Parameters
    ----------
    nx : int, optional
        Number of grid points in x, by default 20.
    ny : int, optional
        Number of grid points in y, by default 20.

    Returns
    -------
    xr.Dataset
        A target rectilinear grid with bounds.
    """
    lon = np.linspace(-25, 25, nx)
    lat = np.linspace(25, 55, ny)

    def get_1d_bnds(arr: np.ndarray) -> np.ndarray:
        """Compute 1D bounds.

        Parameters
        ----------
        arr : np.ndarray
            Center coordinates.

        Returns
        -------
        np.ndarray
            Bounds (n, 2).
        """
        res = np.diff(arr).mean()
        return np.stack([arr - res / 2, arr + res / 2], axis=-1)

    ds = xr.Dataset(
        coords={
            "lat": (("lat",), lat, {"units": "degrees_north", "bounds": "lat_b"}),
            "lon": (("lon",), lon, {"units": "degrees_east", "bounds": "lon_b"}),
            "lat_b": (("lat", "nv"), get_1d_bnds(lat)),
            "lon_b": (("lon", "nv"), get_1d_bnds(lon)),
        }
    )
    return ds


def main() -> None:
    """Main execution function for regridding comparison."""
    # 1. Logic: Perform regridding across multiple methods
    ds_source = create_source_grid()
    ds_target = create_target_grid()

    methods = ["linear", "nearest", "conservative", "bilinear", "cubic"]
    results: dict[str, Any] = {}

    for m in methods:
        print(f"Regridding using method: {m}")
        if m == "linear":
            results[m] = ds_source.regrid.linear(ds_target)
        elif m == "nearest":
            results[m] = ds_source.regrid.nearest(ds_target)
        elif m == "conservative":
            results[m] = ds_source.regrid.conservative(ds_target)
        elif m == "bilinear":
            results[m] = ds_source.regrid.bilinear(ds_target)
        elif m == "cubic":
            results[m] = ds_source.regrid.cubic(ds_target)

    # 2. UI: Track A (Publication-ready)
    print("Generating Track A plot...")
    fig = plt.figure(figsize=(20, 10))
    proj = ccrs.PlateCarree()

    # 1. Source
    ax0 = fig.add_subplot(2, 3, 1, projection=proj)
    ds_source["var"].plot(ax=ax0, x="lon", y="lat", transform=proj, cmap="viridis")
    ax0.set_title("Source (Curvilinear)")
    ax0.coastlines()

    # 2-6. Methods
    for i, m in enumerate(methods, start=2):
        ax = fig.add_subplot(2, 3, i, projection=proj)
        results[m]["var"].plot(ax=ax, x="lon", y="lat", transform=proj, cmap="viridis")
        ax.set_title(f"Method: {m.capitalize()}")
        ax.coastlines()

    plt.tight_layout()
    plt.savefig("all_curvilinear_methods.png", dpi=200)
    print("Saved comparison plot to all_curvilinear_methods.png")

    # 3. UI: Track B (Interactive visualization)
    print("Generating Track B plots (hvplot)...")
    # For demonstration, we just create the interactive objects.
    # To view: hvplot.show(interactive_plot)
    _interactive_plots = {}
    for m in methods:
        _interactive_plots[m] = results[m]["var"].hvplot.quadmesh(
            x="lon",
            y="lat",
            rasterize=True,
            geo=True,
            cmap="viridis",
            title=f"Interactive Curvilinear Regrid ({m})",
            coastline=True,
        )
    print("Track B objects created successfully.")


if __name__ == "__main__":
    main()
