
import numpy as np
import pytest
import xarray as xr
import dask.array as da
from monet_regrid.curvilinear import CurvilinearInterpolator

def create_synthetic_curvilinear(ny, nx, dask=False):
    lon_np, lat_np = np.meshgrid(np.linspace(-20, 20, nx), np.linspace(-20, 20, ny))
    # Add some curvature
    lon_np = lon_np + 2 * np.sin(np.deg2rad(lat_np))
    lat_np = lat_np + 2 * np.cos(np.deg2rad(lon_np))

    if dask:
        lon = da.from_array(lon_np, chunks=(ny//2, nx//2))
        lat = da.from_array(lat_np, chunks=(ny//2, nx//2))
    else:
        lon = lon_np
        lat = lat_np

    ds = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        }
    )
    return ds

@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_curvilinear_eager_lazy_identical(method):
    ny, nx = 50, 50
    source_eager = create_synthetic_curvilinear(ny, nx, dask=False)
    target_eager = create_synthetic_curvilinear(ny, nx, dask=False)

    # Create data to interpolate
    data_np = np.random.rand(ny, nx)
    da_eager = xr.DataArray(data_np, dims=["y", "x"], coords=source_eager.coords, name="test")

    # Eager interpolation
    interp_eager = CurvilinearInterpolator(
        source_eager, target_eager, "lat", "lon", "lat", "lon", method=method
    )
    res_eager = interp_eager(da_eager)

    # Lazy interpolation
    source_lazy = create_synthetic_curvilinear(ny, nx, dask=True)
    target_lazy = create_synthetic_curvilinear(ny, nx, dask=True)
    da_lazy = xr.DataArray(da.from_array(data_np, chunks=(ny//2, nx//2)),
                           dims=["y", "x"], coords=source_lazy.coords, name="test")

    interp_lazy = CurvilinearInterpolator(
        source_lazy, target_lazy, "lat", "lon", "lat", "lon", method=method
    )
    res_lazy = interp_lazy(da_lazy)

    # Check that res_lazy is actually lazy
    assert isinstance(res_lazy.data, da.Array)

    # Compute and compare
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values, rtol=1e-5, atol=1e-5)

def test_coordinate_transformer_dask():
    from monet_regrid.coordinate_transformer import CoordinateTransformer
    ct = CoordinateTransformer()

    lon_np = np.linspace(-180, 180, 100)
    lat_np = np.linspace(-90, 90, 100)

    lon_da = da.from_array(lon_np, chunks=50)
    lat_da = da.from_array(lat_np, chunks=50)

    x_da, y_da, z_da = ct.transform_coordinates(lon_da, lat_da)

    assert isinstance(x_da, da.Array)

    x_np, y_np, z_np = ct.transform_coordinates(lon_np, lat_np)

    np.testing.assert_allclose(x_da.compute(), x_np)
    np.testing.assert_allclose(y_da.compute(), y_np)
    np.testing.assert_allclose(z_da.compute(), z_np)

def test_curvilinear_linear_accuracy():
    """Verify that linear interpolation is accurate for a linear field."""
    ny, nx = 100, 100
    # source grid
    source_ds = create_synthetic_curvilinear(ny, nx)

    # Simple linear field: f(lat, lon) = lat + lon
    data = source_ds["lat"] + source_ds["lon"]

    # target grid (shifted)
    target_ds = create_synthetic_curvilinear(ny, nx)
    target_ds["lat"] = target_ds["lat"] + 0.1
    target_ds["lon"] = target_ds["lon"] + 0.1

    expected = target_ds["lat"] + target_ds["lon"]

    interp = CurvilinearInterpolator(
        source_ds, target_ds, "lat", "lon", "lat", "lon", method="linear"
    )
    result = interp(data)

    # Linear interpolation should be very accurate for a linear field
    inner = slice(2, -2)
    np.testing.assert_allclose(result.values[inner, inner], expected.values[inner, inner], rtol=1e-3, atol=1e-3)
