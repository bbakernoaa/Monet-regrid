
import time
import numpy as np
import xarray as xr
from monet_regrid.curvilinear import CurvilinearInterpolator

def benchmark_na_conus():
    # Source: North America 12km (~0.1 degree)
    # 20N-70N, 170W-50W
    ny_src, nx_src = 500, 1200
    print(f"Creating source grid (North America 12km): {ny_src}x{nx_src} ({ny_src*nx_src/1e6:.1f}M points)")

    # Simulate rotated pole by adding some distortion
    lon_src, lat_src = np.meshgrid(np.linspace(-170, -50, nx_src), np.linspace(20, 70, ny_src))
    # A bit of rotation simulation
    lon_src_dist = lon_src + 2 * np.sin(np.deg2rad(lat_src))
    lat_src_dist = lat_src + 2 * np.cos(np.deg2rad(lon_src))

    source_grid = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat_src_dist),
            "lon": (("y", "x"), lon_src_dist),
        }
    )

    # Target: CONUS 3km (~0.027 degree)
    # 25N-50N, 125W-65W
    ny_tgt, nx_tgt = 900, 2200
    print(f"Creating target grid (CONUS 3km): {ny_tgt}x{nx_tgt} ({ny_tgt*nx_tgt/1e6:.1f}M points)")

    # Rectilinear target
    lon_tgt = np.linspace(-125, -65, nx_tgt)
    lat_tgt = np.linspace(25, 50, ny_tgt)

    target_grid = xr.Dataset(
        coords={
            "lat": (("lat",), lat_tgt),
            "lon": (("lon",), lon_tgt),
        }
    )

    # Create test data
    data = xr.DataArray(
        np.random.rand(ny_src, nx_src),
        dims=["y", "x"],
        coords=source_grid.coords,
        name="variable"
    )

    methods = ["nearest", "linear"]

    for method in methods:
        print(f"\n--- Method: {method} ---")

        start_weights = time.time()
        interpolator = CurvilinearInterpolator(
            source_grid=source_grid,
            target_grid=target_grid,
            source_lat_name="lat",
            source_lon_name="lon",
            target_lat_name="lat",
            target_lon_name="lon",
            method=method
        )
        end_weights = time.time()
        print(f"Weight generation took: {end_weights - start_weights:.2f} seconds")

        start_apply = time.time()
        result = interpolator(data)
        # Trigger computation if it were lazy, but here it's eager.
        # Actually interpolator(data) for eager data is already computed.
        end_apply = time.time()
        print(f"Applying weights took: {end_apply - start_apply:.2f} seconds")
        print(f"Result shape: {result.shape}")

if __name__ == "__main__":
    benchmark_na_conus()
