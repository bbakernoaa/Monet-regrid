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

from monet_regrid.core import CurvilinearRegridder


class MockRegridder(CurvilinearRegridder):
    """A mock class for testing protected methods without a full setup."""

    def __init__(self, source_data, target_grid=None):
        """Bypass the full parent __init__."""
        if target_grid is None:
            target_grid = xr.Dataset(coords={"lat": (("y",), [0.5]), "lon": (("x",), [0.5])})
        self.source_data = source_data
        self.target_grid = target_grid


def test_curvilinear_regridder_lazy_coordinate_generation():
    """
    Test that the fallback coordinate generation is lazy for Dask-backed data.

    This test verifies that when a ``CurvilinearRegridder`` is initialized
    with an ``xarray.DataArray`` that is backed by a Dask array but has no
    explicit coordinates, the internal ``_create_source_grid_from_data``
    method generates lazy (Dask-backed) coordinates instead of eagerly
    computing them. This is critical for performance and memory management.
    """
    # 1. The Logic (Setup)
    # Create a Dask-backed DataArray without explicit coordinates.
    # This simulates a common scenario in lazy data processing pipelines.
    y_size, x_size = 10, 20
    y_chunks, x_chunks = 5, 10
    lazy_data = da.random.random((y_size, x_size), chunks=(y_chunks, x_chunks))
    source_da = xr.DataArray(lazy_data, dims=["y", "x"])

    regridder = MockRegridder(source_data=source_da)

    # 2. The Proof (Execution)
    # Invoke the method responsible for coordinate generation.
    source_grid = regridder._create_source_grid_from_data(source_da)

    # 3. The UI (Verification)
    # Check that the generated coordinates are Dask arrays (lazy).
    assert "latitude" in source_grid.coords
    assert "longitude" in source_grid.coords
    assert isinstance(source_grid["latitude"].data, da.Array)
    assert isinstance(source_grid["longitude"].data, da.Array)

    # Verify that the chunking of the coordinates matches the data's chunking
    # along the corresponding dimensions.
    assert source_grid["latitude"].chunks[0] == source_da.chunks[0]
    assert source_grid["longitude"].chunks[1] == source_da.chunks[1]

    # Verify that the computed coordinate values are correct by creating an
    # expected xr.Dataset and comparing.
    y_coords = np.linspace(0, y_size - 1, y_size)
    x_coords = np.linspace(0, x_size - 1, x_size)
    expected_lon_2d, expected_lat_2d = np.meshgrid(x_coords, y_coords)

    expected_grid = xr.Dataset(
        coords={
            "latitude": (("y", "x"), expected_lat_2d),
            "longitude": (("y", "x"), expected_lon_2d),
        }
    )

    # Use compute on the generated grid for a fair comparison of values
    computed_source_grid = source_grid.compute()
    xr.testing.assert_allclose(computed_source_grid, expected_grid)


def test_curvilinear_regridder_lazy_coordinate_generation_from_numpy():
    """
    Test that the fallback coordinate generation is lazy for NumPy-backed data.

    This test ensures that when a ``CurvilinearRegridder`` is initialized
    with an ``xarray.DataArray`` backed by a NumPy array (eager) but without
    explicit coordinates, the ``_create_source_grid_from_data`` method still
    generates lazy Dask-backed coordinates. This confirms that the regridder
    promotes lazy evaluation even when the input data is in-memory.
    """
    # 1. The Logic (Setup)
    # Create a NumPy-backed DataArray without explicit coordinates.
    y_size, x_size = 10, 20
    eager_data = np.random.random((y_size, x_size))
    source_da = xr.DataArray(eager_data, dims=["y", "x"])

    regridder = MockRegridder(source_data=source_da)

    # 2. The Proof (Execution)
    # Invoke the method responsible for coordinate generation.
    source_grid = regridder._create_source_grid_from_data(source_da)

    # 3. The UI (Verification)
    # Check that the generated coordinates are Dask arrays (lazy), even though
    # the input was a NumPy array.
    assert "latitude" in source_grid.coords
    assert "longitude" in source_grid.coords
    assert isinstance(source_grid["latitude"].data, da.Array)
    assert isinstance(source_grid["longitude"].data, da.Array)

    # Verify that the computed coordinate values are correct.
    y_coords = np.linspace(0, y_size - 1, y_size)
    x_coords = np.linspace(0, x_size - 1, x_size)
    expected_lon_2d, expected_lat_2d = np.meshgrid(x_coords, y_coords)

    expected_grid = xr.Dataset(
        coords={
            "latitude": (("y", "x"), expected_lat_2d),
            "longitude": (("y", "x"), expected_lon_2d),
        }
    )

    # Use compute on the generated grid for a fair comparison of values
    computed_source_grid = source_grid.compute()
    xr.testing.assert_allclose(computed_source_grid, expected_grid)
