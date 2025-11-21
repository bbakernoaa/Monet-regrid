#!/usr/bin/env python
"""
Benchmark script to test performance improvements in rectilinear regridding.
"""

import time

import numpy as np
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def create_test_grids(nx_source=100, ny_source=50, nx_target=50, ny_target=25):
    """Create test source and target grids."""
    # Create source grid
    lon_1d_source = np.linspace(-180, 180, nx_source)
    lat_1d_source = np.linspace(-90, 90, ny_source)

    source_ds = xr.Dataset(
        {
            "temperature": (
                ("lat", "lon"),
                np.random.random((ny_source, nx_source)).astype(np.float32),
                {"units": "K"}
            )
        },
        coords={
            "lat": (["lat"], lat_1d_source),
            "lon": (["lon"], lon_1d_source)
        }
    )

    # Create target grid
    lon_1d_target = np.linspace(-170, 170, nx_target)
    lat_1d_target = np.linspace(-80, ny_target)

    target_ds = xr.Dataset(
        coords={
            "lat": (["lat"], lat_1d_target),
            "lon": (["lon"], lon_1d_target)
        }
    )

    return source_ds, target_ds


def benchmark_regridding():
    """Benchmark the rectilinear regridding performance."""
    print("Creating test grids...")
    source_ds, target_ds = create_test_grids(nx_source=100, ny_source=50, nx_target=50, ny_target=25)

    print(f"Source grid shape: {source_ds['temperature'].shape}")
    print(f"Target grid shape: {target_ds['lat'].shape}, {target_ds['lon'].shape}")

    # Test linear interpolation
    print("\nTesting linear interpolation...")
    regridder = RectilinearRegridder(
        source_data=source_ds,
        target_grid=target_ds,
        method="linear"
    )

    # First call - this will populate the cache
    start_time = time.time()
    result1 = regridder(source_ds["temperature"])
    first_call_time = time.time() - start_time

    print(f"First call time (with cache setup): {first_call_time:.4f} seconds")
    print(f"Result shape: {result1.shape}")
    print(f"NaN count: {np.sum(np.isnan(result1.values))}")

    # Second call - this should benefit from caching
    start_time = time.time()
    result2 = regridder(source_ds["temperature"])
    second_call_time = time.time() - start_time

    print(f"Second call time (with cache hit): {second_call_time:.4f} seconds")
    print(f"Result shape: {result2.shape}")
    print(f"NaN count: {np.sum(np.isnan(result2.values))}")

    # Compare the results to ensure they're the same
    assert np.allclose(result1.values, result2.values, equal_nan=True), "Results should be identical"

    print("\nPerformance improvement:")
    if first_call_time > 0:
        speedup = first_call_time / second_call_time if second_call_time > 0 else float("inf")
        print(f"Speedup on subsequent calls: {speedup:.2f}x")

    print(f"\nFirst call: {first_call_time:.4f}s")
    print(f"Second call: {second_call_time:.4f}s")


def benchmark_multiple_calls():
    """Test performance with multiple calls to see cache effectiveness."""
    print("\n" + "="*60)
    print("Testing multiple calls to evaluate cache effectiveness...")

    source_ds, target_ds = create_test_grids(nx_source=50, ny_source=30, nx_target=25, ny_target=15)

    regridder = RectilinearRegridder(
        source_data=source_ds,
        target_grid=target_ds,
        method="linear"
    )

    # Warm up the cache
    _ = regridder(source_ds["temperature"])

    # Time multiple calls
    times = []
    for i in range(10):
        start_time = time.time()
        result = regridder(source_ds["temperature"])
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"Call {i+1}: {elapsed:.4f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage time per call: {avg_time:.4f}s (±{std_time:.4f}s)")
    print(f"Min time: {np.min(times):.4f}s")
    print(f"Max time: {np.max(times):.4f}s")


if __name__ == "__main__":
    benchmark_regridding()
    benchmark_multiple_calls()
    print("\nBenchmark completed successfully!")
