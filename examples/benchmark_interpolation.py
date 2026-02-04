import time

import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder
from monet_regrid.interpolation import core


def run_benchmark(method="linear", res_deg=1.0):
    """Run interpolation benchmark on a global grid."""
    print(f"\n--- Benchmarking method: {method} at {res_deg} degree resolution ---")  # noqa: T201

    # 1. Generate Grids
    lat = np.arange(-90, 90 + res_deg, res_deg)
    lon = np.arange(-180, 180 + res_deg, res_deg)

    # Source: Curvilinear
    lon2d, lat2d = np.meshgrid(lon, lat)
    np.random.seed(42)
    lon2d_src = lon2d + np.random.uniform(-0.01, 0.01, lon2d.shape)
    lat2d_src = lat2d + np.random.uniform(-0.01, 0.01, lat2d.shape)

    data = np.random.rand(*lat2d_src.shape)
    source_ds = xr.Dataset({"sample": (("y", "x"), data)}, coords={"lon": (("y", "x"), lon2d_src), "lat": (("y", "x"), lat2d_src)})

    # Target: Treated as curvilinear for this benchmark
    target_lon2d, target_lat2d = np.meshgrid(lon, lat)
    target_ds = xr.Dataset(coords={"lat": (("y", "x"), target_lat2d), "lon": (("y", "x"), target_lon2d)})

    results = {}

    for use_numba in [True, False]:
        mode_str = "Numba" if use_numba else "Vectorized Fallback"
        print(f"Testing {mode_str}...")  # noqa: T201

        from monet_regrid.interpolation import base

        original_has_numba_core = core.HAS_NUMBA
        original_has_numba_base = base.HAS_NUMBA
        core.HAS_NUMBA = use_numba
        base.HAS_NUMBA = use_numba

        try:
            start_build = time.perf_counter()
            regridder = CurvilinearRegridder(source_ds, target_ds, method=method)
            _ = regridder()
            build_time = time.perf_counter() - start_build

            start_apply = time.perf_counter()
            _ = regridder(source_ds.sample)
            apply_time = time.perf_counter() - start_apply

            results[mode_str] = {"build": build_time, "apply": apply_time}
            print(f"  Build: {build_time:.2f}s, Apply: {apply_time:.4f}s")  # noqa: T201

        except Exception as e:
            print(f"  Failed: {e}")  # noqa: T201
            results[mode_str] = {"build": np.nan, "apply": np.nan}
        finally:
            core.HAS_NUMBA = original_has_numba_core
            base.HAS_NUMBA = original_has_numba_base

    return results


if __name__ == "__main__":
    results_all = {}
    results_all["nearest"] = run_benchmark(method="nearest", res_deg=1.0)
    results_all["linear"] = run_benchmark(method="linear", res_deg=1.0)

    print("\n" + "=" * 80)  # noqa: T201
    print(f"{'Method':<15} | {'Path':<20} | {'Build (s)':<15} | {'Apply (s)':<15}")  # noqa: T201
    print("-" * 80)  # noqa: T201
    for method, res in results_all.items():
        for mode, times in res.items():
            print(f"{method:<15} | {mode:<20} | {times['build']:<15.2f} | {times['apply']:<15.4f}")  # noqa: T201
    print("=" * 80)  # noqa: T201
