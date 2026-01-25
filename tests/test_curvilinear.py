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


def test_curvilinear_regridder_linear_structured_accuracy():
    """
    Test the numerical accuracy of the structured linear interpolation.

    This test verifies that the Numba-accelerated grid search for linear
    interpolation is numerically correct. It uses a known, simple linear
    function as the ground truth. Linear interpolation of a linear field
    should be nearly exact.
    """
    # 1. The Logic (Setup)
    # Create a slightly irregular 2D source grid.
    y_coords = np.linspace(30, 40, 50)
    x_coords = np.linspace(-120, -110, 100)
    xx, yy = np.meshgrid(x_coords, y_coords)
    source_lon_2d = xx + 0.1 * np.sin(np.deg2rad(yy))
    source_lat_2d = yy + 0.1 * np.sin(np.deg2rad(xx))

    source_ds = xr.Dataset(
        coords={
            "latitude": (("y", "x"), source_lat_2d),
            "longitude": (("y", "x"), source_lon_2d),
        }
    )

    # Create source data from a simple analytical function.
    def analytical_func(lat, lon):
        """A simple linear function for testing interpolation accuracy."""
        return 2 * lat + 3 * lon

    source_da = analytical_func(source_ds["latitude"], source_ds["longitude"])
    source_da.name = "temperature"

    # Create a regular target grid.
    target_ds = xr.Dataset(
        coords={
            "lat": np.linspace(32, 38, 20),
            "lon": np.linspace(-118, -112, 30),
        }
    )

    # 2. The Proof (Execution)
    # Initialize the regridder, which triggers the Numba-accelerated path.
    regridder = CurvilinearRegridder(source_ds, target_ds, method="linear")

    # Perform the interpolation.
    interpolated_da = regridder(source_da)

    # 3. The UI (Verification)
    # Calculate the expected values on the target grid using the analytical function.
    expected_lon_2d, expected_lat_2d = np.meshgrid(target_ds["lon"], target_ds["lat"])
    expected_values = analytical_func(expected_lat_2d, expected_lon_2d)
    expected_da = xr.DataArray(
        expected_values,
        coords={"lat": target_ds["lat"], "lon": target_ds["lon"]},
        dims=["lat", "lon"],
        name="temperature",
    )

    # Assert that the interpolated data is numerically very close to the expected data.
    # A tight tolerance is used because the interpolation should be nearly exact.
    xr.testing.assert_allclose(interpolated_da, expected_da, rtol=1e-6)


def test_curvilinear_regridder_hybrid_accuracy():
    """
    Test the numerical accuracy of the hybrid (polar/non-polar) interpolation.

    This test is critical for verifying that the merge logic for the hybrid
    interpolation strategy is correct. It creates a target grid that spans
    the polar boundary (85 degrees latitude), forcing the use of both the fast
    Numba path and the robust Delaunay fallback path. The final, merged result
    is then compared against a known analytical function to ensure its
    numerical integrity.
    """
    # 1. The Logic (Setup)
    # Create a high-latitude source grid.
    y_coords = np.linspace(80, 90, 20)
    x_coords = np.linspace(-180, 180, 40)
    source_lon_2d, source_lat_2d = np.meshgrid(x_coords, y_coords)

    source_ds = xr.Dataset(
        coords={
            "latitude": (("y", "x"), source_lat_2d),
            "longitude": (("y", "x"), source_lon_2d),
        }
    )

    # Create source data from a simple analytical function.
    def analytical_func(lat, lon):
        """A simple linear function for testing interpolation accuracy."""
        return 2 * lat + 0.5 * lon

    source_da = analytical_func(source_ds["latitude"], source_ds["longitude"])
    source_da.name = "polar_temp"

    # Create a target grid that straddles the 85-degree polar threshold.
    target_lat = np.linspace(84, 86, 10)
    target_lon = np.linspace(-170, 170, 20)
    target_ds = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

    # 2. The Proof (Execution)
    # Initialize the regridder, which will trigger the hybrid logic.
    regridder = CurvilinearRegridder(source_ds, target_ds, method="linear")
    interpolated_da = regridder(source_da)

    # 3. The UI (Verification)
    # Calculate the expected values on the target grid using the analytical function.
    expected_lon_2d, expected_lat_2d = np.meshgrid(target_lon, target_lat)
    expected_values = analytical_func(expected_lat_2d, expected_lon_2d)
    expected_da = xr.DataArray(
        expected_values,
        coords={"lat": target_lat, "lon": target_lon},
        dims=["lat", "lon"],
        name="polar_temp",
    )

    # Assert that the interpolated data is numerically very close to the expected data.
    # A slightly higher tolerance is used to account for the less precise (but more
    # robust) Delaunay triangulation used in the polar region fallback.
    xr.testing.assert_allclose(interpolated_da, expected_da, rtol=1e-2)
