from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def test_rectilinear_lazy_coordinate_generation():
    """Test that RectilinearRegridder can handle data without explicit coordinates."""
    # Create data WITHOUT coordinates, only dimensions
    # Using dask to ensure it stays lazy
    # We use matching names for target to avoid expansion
    data_values = da.random.random((10, 20), chunks=(5, 5))
    data = xr.DataArray(data_values, dims=["lat", "lon"])

    target_grid = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(0, 9, 5)),
            "lon": (("lon",), np.linspace(0, 19, 10)),
        }
    )

    # RectilinearRegridder should now accept this
    regridder = RectilinearRegridder(data, target_grid, method="linear")

    # Execute regridding
    regridded = regridder()

    # Verify result
    assert regridded.shape == (5, 10)
    assert "lat" in regridded.coords
    assert "lon" in regridded.coords
    assert "history" in regridded.attrs
    assert "Regridded using RectilinearRegridder" in regridded.attrs["history"]


def test_identify_cf_coordinates_standard():
    """Test that identify_cf_coordinates finds coordinates with standard names."""
    from monet_regrid.utils import identify_cf_coordinates

    # Standard names in coords
    ds = xr.Dataset(
        coords={"lat": (("lat",), np.arange(10)), "lon": (("lon",), np.arange(10))},
        data_vars={"a": (("lat", "lon"), np.zeros((10, 10)))},
    )
    lat_name, lon_name = identify_cf_coordinates(ds)
    assert lat_name == "lat"
    assert lon_name == "lon"
