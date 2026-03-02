"""
Tests for Dask-related optimizations in monet-regrid.

This file is part of monet-regrid.
This derivative work Copyright (c) 2024 monet-regrid Developers.
"""

import dask.array as da
import xarray as xr

from monet_regrid import utils
from monet_regrid.constants import GridType


def test_lazy_coordinate_generation():
    """
    Test that the fallback coordinate generation is lazy for Dask-backed data.

    This test validates that when a CurvilinearRegridder is initialized with
    a Dask-backed DataArray that lacks explicit coordinates, the internally
    generated source grid coordinates are also Dask arrays, preventing eager
    loading.
    """
    # 1. The Logic (Setup)
    # Create a large, Dask-backed DataArray without explicit coordinates
    source_shape = (1000, 2000)
    source_chunks = (500, 500)
    source_data = da.ones(source_shape, chunks=source_chunks)
    source_da = xr.DataArray(
        source_data,
        dims=["y", "x"],
        name="temperature",
    )

    # 2. The Proof (Execution & Validation)
    # Use the centralized utility
    source_grid = utils.ensure_spatial_coords(source_da, GridType.CURVILINEAR)

    # Assert that the coordinates are 2D
    assert source_grid["latitude"].ndim == 2, "Latitude should be 2D"
    assert source_grid["longitude"].ndim == 2, "Longitude should be 2D"

    # Assert that the underlying data is a dask array
    assert isinstance(
        source_grid["latitude"].data, da.Array
    ), f"Expected dask.array.Array, got {type(source_grid['latitude'].data)}"
    assert isinstance(
        source_grid["longitude"].data, da.Array
    ), f"Expected dask.array.Array, got {type(source_grid['longitude'].data)}"

    # 3. The UI (CLI Command)
    # The command to run this test would be:
    # python -m pytest tests/test_dask_optimization.py
