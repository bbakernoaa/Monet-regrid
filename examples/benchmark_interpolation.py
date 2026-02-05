import time

import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder
from monet_regrid.interpolation import core


def run_benchmark(method="linear", res_deg=1.0):
    """Run interpolation benchmark comparing monet-regrid and xregrid."""
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

    # Target: Rectilinear
    target_ds = xr.Dataset(coords={"lat": (("lat",), lat), "lon": (("lon",), lon)})

    results = {}

    # 2. Benchmark monet-regrid
    for use_numba in [True, False]:
        mode_str = f"monet-regrid ({'Numba' if use_numba else 'Vectorized Fallback'})"
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

    # 3. Benchmark xregrid (conditionally import to satisfy linter when not present)
    xregrid_method_map = {"linear": "bilinear", "nearest": "nearest_s2d", "conservative": "conservative"}

    if method in xregrid_method_map:
        mode_str = "xregrid"
        print(f"Testing {mode_str}...")  # noqa: T201
        xr_method = xregrid_method_map[method]

        try:
            # We use local import to avoid unused import at top level
            import xregrid as xr_bench  # noqa: F401

            start_build = time.perf_counter()
            # use .regrid.to()
            _ = source_ds.regrid.to(target_ds, method=xr_method)
            total_time = time.perf_counter() - start_build

            results[mode_str] = {"build": total_time, "apply": np.nan}
            print(f"  Total (Build+Apply): {total_time:.2f}s")  # noqa: T201

        except ImportError:
            print("  xregrid not installed, skipping comparison.")  # noqa: T201
            results[mode_str] = {"build": np.nan, "apply": np.nan}
        except Exception as e:
            print(f"  Failed: {e}")  # noqa: T201
            results[mode_str] = {"build": np.nan, "apply": np.nan}

    return results


if __name__ == "__main__":
    results_all = {}
    results_all["nearest"] = run_benchmark(method="nearest", res_deg=1.0)
    results_all["linear"] = run_benchmark(method="linear", res_deg=1.0)

    print("\n" + "=" * 90)  # noqa: T201
    print(f"{'Method':<15} | {'Path':<35} | {'Build (s)':<15} | {'Apply (s)':<15}")  # noqa: T201
    print("-" * 90)  # noqa: T201
    for method, res in results_all.items():
        for mode, times in res.items():
            print(f"{method:<15} | {mode:<35} | {times['build']:<15.2f} | {times['apply']:<15.4f}")  # noqa: T201
    print("=" * 90)  # noqa: T201
