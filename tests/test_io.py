"""
Unit tests for the I/O functions of the monet-regrid library.

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

Modifications: Package renamed from xarray-regrid to monet-regrid,
URLs updated, and documentation adapted for new branding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def test_regridder_io_netcdf(tmp_path: Path) -> None:
    """Test writing and reading a regridder to and from a netCDF file."""
    # Create a dummy regridder
    source_data = xr.Dataset(
        {
            "temp": (
                ("y", "x"),
                np.random.rand(10, 20),
                {"units": "K", "long_name": "temperature"},
            )
        },
        coords={
            "y": np.arange(10),
            "x": np.arange(20),
            "lat": (("y", "x"), np.random.rand(10, 20)),
            "lon": (("y", "x"), np.random.rand(10, 20)),
        },
    )
    target_grid = xr.Dataset(
        coords={
            "y": np.arange(5),
            "x": np.arange(10),
            "lat": (("y", "x"), np.random.rand(5, 10)),
            "lon": (("y", "x"), np.random.rand(5, 10)),
        }
    )
    regridder = CurvilinearRegridder(
        source_data=source_data, target_grid=target_grid, method="linear"
    )

    # Write to netCDF
    filepath = tmp_path / "regridder.nc"
    regridder.to_netcdf(filepath)

    # Read from netCDF
    loaded_regridder = CurvilinearRegridder.from_netcdf(filepath)

    # Check that the loaded regridder is the same as the original
    assert regridder.method == loaded_regridder.method
    assert_allclose(regridder.source_data, loaded_regridder.source_data)
    assert_allclose(regridder.target_grid, loaded_regridder.target_grid)

    # Test with RectilinearRegridder
    source_data = xr.Dataset(
        {
            "temp": (
                ("y", "x"),
                np.random.rand(10, 20),
                {"units": "K", "long_name": "temperature"},
            )
        },
        coords={"y": np.arange(10), "x": np.arange(20)},
    )
    target_grid = xr.Dataset(coords={"y": np.arange(5), "x": np.arange(10)})
    regridder = RectilinearRegridder(
        source_data=source_data, target_grid=target_grid, method="linear"
    )

    # Write to netCDF
    filepath = tmp_path / "regridder_rect.nc"
    regridder.to_netcdf(filepath)

    # Read from netCDF
    loaded_regridder = RectilinearRegridder.from_netcdf(filepath)

    # Check that the loaded regridder is the same as the original
    assert regridder.method == loaded_regridder.method
    assert_allclose(regridder.source_data, loaded_regridder.source_data)
    assert_allclose(regridder.target_grid, loaded_regridder.target_grid)
