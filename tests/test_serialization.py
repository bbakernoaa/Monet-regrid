"""
Unit tests for regridder serialization.

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
def rectilinear_source_grid():
    """Create a rectilinear source grid for testing."""
    return xr.Dataset(
        {
            "air_temperature": (
                ("y", "x"),
                np.arange(12).reshape(3, 4),
            ),
        },
        coords={
            "lat": (
                ("y",),
                np.array([50, 51, 52]),
            ),
            "lon": (
                ("x",),
                np.array([0, 1, 2, 3]),
            ),
        },
    )


@pytest.fixture()
def rectilinear_target_grid():
    """Create a rectilinear target grid for testing."""
    return xr.Dataset(
        coords={
            "lat": (
                ("y_new",),
                np.array([50.5, 51.5]),
            ),
            "lon": (
                ("x_new",),
                np.array([0.5, 1.5, 2.5]),
            ),
        }
    )


@pytest.fixture()
def curvilinear_source_grid():
    """Create a curvilinear source grid for testing."""
    lon = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    lat = np.array(
        [
            [50, 50, 50, 50],
            [51, 51, 51, 51],
            [52, 52, 52, 52],
        ],
    )
    return xr.Dataset(
        {
            "air_temperature": (
                ("y", "x"),
                np.arange(12).reshape(3, 4),
            ),
        },
        coords={
            "lat": (
                ("y", "x"),
                lat,
            ),
            "lon": (
                ("y", "x"),
                lon,
            ),
        },
    )


@pytest.fixture()
def curvilinear_target_grid():
    """Create a curvilinear target grid for testing."""
    return xr.Dataset(
        coords={
            "lat": (
                ("y_new", "x_new"),
                np.array([[50.5, 50.5, 50.5], [51.5, 51.5, 51.5]]),
            ),
            "lon": (
                ("y_new", "x_new"),
                np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]]),
            ),
        }
    )


def test_rectilinear_regridder_serialization(
    rectilinear_source_grid, rectilinear_target_grid, tmp_path
):
    """Test saving and loading a RectilinearRegridder."""
    # Create a regridder and regrid the data
    regridder = RectilinearRegridder(
        source_data=rectilinear_source_grid, target_grid=rectilinear_target_grid
    )
    expected = regridder()

    # Save the regridder
    filepath = tmp_path / "regridder.nc"
    regridder.to_file(filepath)

    # Load the regridder and apply it to the same data
    loaded_regridder = RectilinearRegridder.from_file(filepath)
    result = loaded_regridder(rectilinear_source_grid)

    # Check that the results are identical
    assert_allclose(result, expected)


def test_curvilinear_regridder_serialization(
    curvilinear_source_grid, curvilinear_target_grid, tmp_path
):
    """Test saving and loading a CurvilinearRegridder."""
    # Create a regridder and regrid the data
    regridder = CurvilinearRegridder(
        source_data=curvilinear_source_grid, target_grid=curvilinear_target_grid
    )
    expected = regridder()

    # Save the regridder
    filepath = tmp_path / "regridder.nc"
    regridder.to_file(filepath)

    # Load the regridder and apply it to the same data
    loaded_regridder = CurvilinearRegridder.from_file(filepath)
    result = loaded_regridder(curvilinear_source_grid)

    # Check that the results are identical
    assert_allclose(result, expected)
