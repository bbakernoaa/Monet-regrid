"""
Unit tests for the CurvilinearRegridder.

This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 [Your Organization].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications: Package renamed from xarray-regrid to monet-regrid,
URLs updated, and documentation adapted for new branding.
"""

import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid import utils
from monet_regrid.constants import GridType


def test_curvilinear_regridder_lazy_coordinate_generation():
    """
    Test that the fallback coordinate generation is lazy for Dask-backed data.

    This test verifies that when a ``CurvilinearRegridder`` is called
    with an ``xarray.DataArray`` that is backed by a Dask array but has no
    explicit coordinates, it generates lazy (Dask-backed) coordinates instead
    of eagerly computing them.
    """
    # 1. The Logic (Setup)
    # Create a Dask-backed DataArray without explicit coordinates.
    y_size, x_size = 10, 20
    y_chunks, x_chunks = 5, 10
    lazy_data = da.random.random((y_size, x_size), chunks=(y_chunks, x_chunks))
    source_da = xr.DataArray(lazy_data, dims=["y", "x"])

    # 2. The Proof (Execution)
    # Use the utility directly as it's what the regridder calls
    source_with_coords = utils.ensure_spatial_coords(source_da, GridType.CURVILINEAR)

    # 3. The UI (Verification)
    # Check that the generated coordinates are Dask arrays (lazy).
    assert "latitude" in source_with_coords.coords
    assert "longitude" in source_with_coords.coords
    assert isinstance(source_with_coords["latitude"].data, da.Array)
    assert isinstance(source_with_coords["longitude"].data, da.Array)

    # Verify that the chunking of the coordinates matches the data's chunking
    # along the corresponding dimensions.
    assert source_with_coords["latitude"].chunks[0] == source_da.chunks[0]
    assert source_with_coords["longitude"].chunks[1] == source_da.chunks[1]

    # Verify that the computed coordinate values are correct.
    y_coords = np.linspace(0, y_size - 1, y_size)
    x_coords = np.linspace(0, x_size - 1, x_size)
    # Note: ensure_spatial_coords uses broadcast which is (y[:, np.newaxis], x)
    expected_lat_2d = np.broadcast_to(y_coords[:, np.newaxis], (10, 20))
    expected_lon_2d = np.broadcast_to(x_coords, (10, 20))

    expected_grid = xr.Dataset(
        coords={
            "latitude": (("y", "x"), expected_lat_2d),
            "longitude": (("y", "x"), expected_lon_2d),
        }
    )

    # Use compute on the generated grid for a fair comparison of values
    computed_source_grid = xr.Dataset(
        coords={
            "latitude": source_with_coords["latitude"].compute(),
            "longitude": source_with_coords["longitude"].compute(),
        }
    )
    xr.testing.assert_allclose(computed_source_grid, expected_grid)


def test_curvilinear_interpolator_is_lazy():
    """
    Test that the CurvilinearInterpolator is lazy and only builds when called.
    """
    from monet_regrid.curvilinear import CurvilinearInterpolator

    # 1. The Logic (Setup)
    # Create Dask-backed source and target grids.
    source_da = xr.DataArray(
        da.random.random((10, 20), chunks=(5, 10)),
        dims=["y", "x"],
        coords={
            "lat": (("y", "x"), np.random.uniform(0, 10, size=(10, 20))),
            "lon": (("y", "x"), np.random.uniform(0, 20, size=(10, 20))),
        },
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y_new",), np.arange(0.5, 10, 2)),
            "lon": (("x_new",), np.arange(0.5, 20, 2)),
        }
    )

    # 2. The Proof (Execution & Verification)
    # Instantiate the interpolator.
    interpolator = CurvilinearInterpolator(
        source_grid=source_da.to_dataset(name="data"),
        target_grid=target_ds,
        source_lat_name="lat",
        source_lon_name="lon",
        target_lat_name="lat",
        target_lon_name="lon",
        method="linear",
    )

    # Assert that the engine has not been built yet.
    assert interpolator.interpolation_engine is None
    assert not interpolator._is_built

    # Call the interpolator to trigger the build.
    regridded_da = interpolator(source_da)

    # Assert that the engine has now been built.
    assert interpolator.interpolation_engine is not None
    assert interpolator._is_built

    # 3. The UI (Verification)
    # Check that the output is a Dask-backed DataArray and has the correct shape.
    assert isinstance(regridded_da.data, da.Array)
    assert regridded_da.shape == (5, 10)


def test_curvilinear_regridder_lazy_coordinate_generation_from_numpy():
    """
    Test that the fallback coordinate generation is lazy for NumPy-backed data.

    This test ensures that when a ``CurvilinearRegridder`` is called
    with an ``xarray.DataArray`` backed by a NumPy array (eager) but without
    explicit coordinates, it still generates lazy Dask-backed coordinates.
    """
    # 1. The Logic (Setup)
    # Create a NumPy-backed DataArray without explicit coordinates.
    y_size, x_size = 10, 20
    eager_data = np.random.random((y_size, x_size))
    source_da = xr.DataArray(eager_data, dims=["y", "x"])

    # 2. The Proof (Execution)
    source_with_coords = utils.ensure_spatial_coords(source_da, GridType.CURVILINEAR)

    # 3. The UI (Verification)
    # Check that the generated coordinates are Dask arrays (lazy).
    assert "latitude" in source_with_coords.coords
    assert "longitude" in source_with_coords.coords
    assert isinstance(source_with_coords["latitude"].data, da.Array)
    assert isinstance(source_with_coords["longitude"].data, da.Array)

    # Verify that the computed coordinate values are correct.
    y_coords = np.linspace(0, y_size - 1, y_size)
    x_coords = np.linspace(0, x_size - 1, x_size)
    expected_lat_2d = np.broadcast_to(y_coords[:, np.newaxis], (10, 20))
    expected_lon_2d = np.broadcast_to(x_coords, (10, 20))

    expected_grid = xr.Dataset(
        coords={
            "latitude": (("y", "x"), expected_lat_2d),
            "longitude": (("y", "x"), expected_lon_2d),
        }
    )

    # Use compute on the generated grid for a fair comparison of values
    computed_source_grid = xr.Dataset(
        coords={
            "latitude": source_with_coords["latitude"].compute(),
            "longitude": source_with_coords["longitude"].compute(),
        }
    )
    xr.testing.assert_allclose(computed_source_grid, expected_grid)
