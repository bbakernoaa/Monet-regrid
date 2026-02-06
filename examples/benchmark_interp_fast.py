"""
Benchmark and visualization script for vectorized rectilinear interpolation.

Compares the performance of the optimized fast path against standard xarray.interp.
Follows the Aero Protocol for visualization and documentation.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None
import xarray as xr

from monet_regrid.methods.interp import interp_regrid


def benchmark() -> None:
    """Benchmark the vectorized interpolation against standard xarray.interp.

    This function creates a synthetic multidimensional dataset, performs
    regridding using both the optimized fast path and the standard xarray path,
    measures their performance, and optionally visualizes the result.

    Returns
    -------
    None

    Examples
    --------
    >>> benchmark()
    Benchmarking interpolation...
    Optimized Path Time: ...
    Standard Path Time: ...
    Speedup: ...
    """
    # 1. Setup Data: (time, lat, lon)
    # 50 time steps, 200x400 grid
    ntime, nlat, nlon = 50, 200, 400
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)
    time_coords = np.arange(ntime)

    da = xr.DataArray(
        np.random.rand(ntime, nlat, nlon),
        dims=["time", "lat", "lon"],
        coords={"time": time_coords, "lat": lat, "lon": lon},
        name="temperature",
    )

    # Target grid (upsampling)
    nlat_new, nlon_new = 150, 300
    target_ds = xr.Dataset(coords={"lat": np.linspace(-90, 90, nlat_new), "lon": np.linspace(-180, 180, nlon_new)})

    print(f"Benchmarking interpolation of {da.shape} -> ({ntime}, {nlat_new}, {nlon_new})")  # noqa: T201

    # 2. Optimized Path (monet-regrid vectorized)
    start = time.perf_counter()
    res_fast = interp_regrid(da, target_ds, method="linear")
    fast_time = time.perf_counter() - start
    print(f"Optimized Path Time: {fast_time:.4f}s")  # noqa: T201

    # 3. Standard Path (xarray.interp fallback)
    start = time.perf_counter()
    _res_slow = da.interp(lat=target_ds.lat, lon=target_ds.lon, method="linear")
    slow_time = time.perf_counter() - start
    print(f"Standard Path Time: {slow_time:.4f}s")  # noqa: T201

    speedup = slow_time / fast_time
    print(f"Speedup: {speedup:.2f}x")  # noqa: T201

    # 4. Visualization (Track A: Static)
    if ccrs:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"projection": ccrs.PlateCarree()})

        # Plot first time step
        da.isel(time=0).plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap="viridis")
        axes[0].coastlines()
        axes[0].set_title("Original Grid (t=0)")

        res_fast.isel(time=0).plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap="viridis")
        axes[1].coastlines()
        axes[1].set_title(f"Regridded Grid (t=0)\nSpeedup: {speedup:.1f}x")

        plt.tight_layout()
        plt.savefig("interp_benchmark_results.png", dpi=150)
        print("Benchmark plot saved to 'interp_benchmark_results.png'")  # noqa: T201
    else:
        print("Cartopy not installed, skipping plot generation.")  # noqa: T201


if __name__ == "__main__":
    benchmark()
