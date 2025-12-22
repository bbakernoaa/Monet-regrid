"""
Tests for the serialization of regridder objects.

This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 monet-regrid Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

import monet_regrid  # noqa: F401
from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


@pytest.fixture()
def source_data_rectilinear():
    """Create a dummy source dataset with rectilinear grid."""
    return xr.Dataset(
        {"var1": (("y", "x"), np.random.rand(10, 20))},
        coords={
            "lat": (("y",), np.linspace(0, 1, 10)),
            "lon": (("x",), np.linspace(0, 1, 20)),
        },
    )


@pytest.fixture()
def target_grid_rectilinear():
    """Create a dummy target dataset with rectilinear grid."""
    return xr.Dataset(
        coords={
            "lat": (("y_new",), np.linspace(0, 1, 5)),
            "lon": (("x_new",), np.linspace(0, 1, 10)),
        }
    )


@pytest.fixture()
def source_data_curvilinear():
    """Create a dummy source dataset with curvilinear grid."""
    lat = np.linspace(0, 1, 10)
    lon = np.linspace(0, 1, 20)
    lon, lat = np.meshgrid(lon, lat)
    return xr.Dataset(
        {"var1": (("y", "x"), np.random.rand(10, 20))},
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )


@pytest.fixture()
def target_grid_curvilinear():
    """Create a dummy target dataset with curvilinear grid."""
    lat = np.linspace(0, 1, 5)
    lon = np.linspace(0, 1, 10)
    lon, lat = np.meshgrid(lon, lat)
    return xr.Dataset(
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )


def test_rectilinear_regridder_serialization(
    tmp_path, source_data_rectilinear, target_grid_rectilinear
):
    """Test saving and loading a RectilinearRegridder."""
    regridder = RectilinearRegridder(
        source_data=source_data_rectilinear,
        target_grid=target_grid_rectilinear,
        method="linear",
    )
    filepath = tmp_path / "regridder.nc"
    regridder.to_file(filepath)

    loaded_regridder = RectilinearRegridder.from_file(filepath)

    # Check that the regridders are the same
    assert regridder.method == loaded_regridder.method
    assert regridder.time_dim == loaded_regridder.time_dim
    assert regridder.method_kwargs == loaded_regridder.method_kwargs
    assert_allclose(regridder.source_data, loaded_regridder.source_data)
    assert_allclose(regridder.target_grid, loaded_regridder.target_grid)

    # Check that they produce the same result
    expected = regridder()
    actual = loaded_regridder()
    assert_allclose(expected, actual)


def test_curvilinear_regridder_serialization(
    tmp_path, source_data_curvilinear, target_grid_curvilinear
):
    """Test saving and loading a CurvilinearRegridder."""
    regridder = CurvilinearRegridder(
        source_data=source_data_curvilinear,
        target_grid=target_grid_curvilinear,
        method="linear",
    )
    filepath = tmp_path / "regridder.nc"
    regridder.to_file(filepath)

    loaded_regridder = CurvilinearRegridder.from_file(filepath)

    # Check that the regridders are the same
    assert regridder.method == loaded_regridder.method
    assert regridder.method_kwargs == loaded_regridder.method_kwargs
    assert_allclose(regridder.source_data, loaded_regridder.source_data)
    assert_allclose(regridder.target_grid, loaded_regridder.target_grid)

    # Check that they produce the same result
    expected = regridder()
    actual = loaded_regridder()
    assert_allclose(expected, actual)
