"""High-resolution stress test using Dask."""

import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator


def test_high_resolution_stress_dask():
    """Test high-resolution interpolation with Dask arrays."""
    # Try to use distributed client if available to simulate cluster behavior
    try:
        from dask.distributed import Client  # noqa: PLC0415

        # Start a local cluster
        client = Client(dashboard_address=None)
    except ImportError:
        client = None

    # Create large source grid (e.g., 500x500)
    # Using smaller size for CI/local testing to avoid OOM, but large enough to stress
    # Reduce size slightly to avoid large graph warning (2.46 GiB graph for 10000x10000 with (50,50) chunks)
    ny, nx = 5000, 5000
    source_x, source_y = np.meshgrid(np.arange(nx), np.arange(ny))
    source_lat = 30 + 0.05 * source_x
    source_lon = -100 + 0.05 * source_y

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    # Create target grid
    target_ny, target_nx = 2000, 2000
    target_x, target_y = np.meshgrid(np.linspace(0, nx - 1, target_nx), np.linspace(0, ny - 1, target_ny))
    target_lat = 30 + 0.05 * target_x
    target_lon = -100 + 0.05 * target_y

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create large dask array data
    # Increase chunk size to reduce graph size (chunks=(50,50) creates 40000 chunks for 2000x2000)
    # Using larger chunks is better practice
    data_dask = da.random.random((ny, nx), chunks=(200, 200))
    test_data = xr.DataArray(data_dask, dims=["y", "x"])

    # Interpolate using nearest neighbor (efficient for large grids)
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")

    # This should be lazy
    result = interpolator(test_data)

    assert isinstance(result.data, da.Array)

    # Compute result
    computed_result = result.compute()

    assert computed_result.shape == (target_ny, target_nx)
    assert not np.isnan(computed_result.values).all()

    if client:
        client.close()


if __name__ == "__main__":
    test_high_resolution_stress_dask()
