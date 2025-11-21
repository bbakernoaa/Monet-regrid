#!/usr/bin/env python
"""
Benchmark script to test performance improvements in curvilinear regridding.
"""

import time

import numpy as np
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator


def create_test_grids(nx_source=100, ny_source=50, nx_target=50, ny_target=25):
    """Create test source and target grids."""
    # Create source grid
    lon_1d_source = np.linspace(-180, 180, nx_source)
    lat_1d_source = np.linspace(-90, 90, ny_source)
    source_lon_2d, source_lat_2d = np.meshgrid(lon_1d_source, lat_1d_source)

    # Add some curvilinear distortion
    source_lon_2d = source_lon_2d + 0.5 * np.sin(np.radians(source_lat_2d)) * np.cos(np.radians(source_lon_2d))
    source_lat_2d = source_lat_2d + 0.3 * np.cos(np.radians(source_lat_2d)) * np.sin(np.radians(source_lon_2d))

    source_ds = xr.Dataset(
        {
            "temperature": (
                ("y", "x"),
                np.random.random((ny_source, nx_source)).astype(np.float32),
                {"units": "K"}
            )
        },
        coords={
            "lon": (("y", "x"), source_lon_2d),
            "lat": (("y", "x"), source_lat_2d)
        }
    )

    # Create target grid
    lon_1d_target = np.linspace(-180, 180, nx_target)
    lat_1d_target = np.linspace(-90, ny_target)
    target_lon_2d, target_lat_2d = np.meshgrid(lon_1d_target, lat_1d_target)

    # Add some curvilinear distortion
    target_lon_2d = target_lon_2d + 0.3 * np.sin(np.radians(target_lat_2d)) * np.cos(np.radians(target_lon_2d))
    target_lat_2d = target_lat_2d + 0.2 * np.cos(np.radians(target_lat_2d)) * np.sin(np.radians(target_lon_2d))

    target_ds = xr.Dataset(
        coords={
            "lon": (("y", "x"), target_lon_2d),
            "lat": (("y", "x"), target_lat_2d)
        }
    )

    return source_ds, target_ds


def benchmark_interpolation():
    """Benchmark the curvilinear interpolation performance."""
    print("Creating test grids...")
    source_ds, target_ds = create_test_grids(nx_source=100, ny_source=50, nx_target=50, ny_target=25)

    print(f"Source grid shape: {source_ds['temperature'].shape}")
    print(f"Target grid shape: {target_ds['lon'].shape}")

    # Test linear interpolation
    print("\nTesting linear interpolation...")
    interpolator_linear = CurvilinearInterpolator(
        source_grid=source_ds,
        target_grid=target_ds,
        method="linear"
    )

    start_time = time.time()
    result_linear = interpolator_linear(source_ds["temperature"])
    linear_elapsed = time.time() - start_time

    print(f"Linear interpolation time: {linear_elapsed:.4f} seconds")
    print(f"Result shape: {result_linear.shape}")
    print(f"NaN count: {np.sum(np.isnan(result_linear.values))}")

    # Test nearest neighbor interpolation
    print("\nTesting nearest neighbor interpolation...")
    interpolator_nearest = CurvilinearInterpolator(
        source_grid=source_ds,
        target_grid=target_ds,
        method="nearest"
    )

    start_time = time.time()
    result_nearest = interpolator_nearest(source_ds["temperature"])
    nearest_elapsed = time.time() - start_time

    print(f"Nearest neighbor interpolation time: {nearest_elapsed:.4f} seconds")
    print(f"Result shape: {result_nearest.shape}")
    print(f"NaN count: {np.sum(np.isnan(result_nearest.values))}")

    print("\nPerformance summary:")
    print(f"Linear interpolation: {linear_elapsed:.4f}s")
    print(f"Nearest neighbor interpolation: {nearest_elapsed:.4f}s")


if __name__ == "__main__":
    benchmark_interpolation()
