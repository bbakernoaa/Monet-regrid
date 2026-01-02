import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def test_rectilinear_regridder_instantiation_and_call():
    """
    Test direct instantiation and usage of the RectilinearRegridder class.

    This test provides "Proof" for the refactoring of the class docstrings
    and type hints by ensuring the core functionality remains correct.
    """
    # 1. The Logic (Setup)
    source_da = xr.DataArray(
        np.arange(12).reshape(4, 3),
        dims=["lat", "lon"],
        coords={
            "lat": np.array([40, 41, 42, 43]),
            "lon": np.array([-100, -99, -98]),
        },
    )
    target_ds = xr.Dataset().assign_coords(
        {
            "lat": (("y_new",), np.arange(40.5, 43, 1)),
            "lon": (("x_new",), np.arange(-99.5, -98, 1)),
        }
    )

    # 2. The Action
    regridder = RectilinearRegridder(source_data=source_da, target_grid=target_ds, method="linear")
    regridded_da = regridder()

    # 3. The Proof (Validation)
    assert isinstance(regridded_da, xr.DataArray)
    assert regridded_da.shape == (3, 2)
    assert "lat" in regridded_da.coords
    assert "lon" in regridded_da.coords
    np.testing.assert_allclose(regridded_da["lat"], np.array([40.5, 41.5, 42.5]))
    np.testing.assert_allclose(regridded_da["lon"], np.array([-99.5, -98.5]))

    # Check a value to confirm interpolation is working as expected
    expected_value = 2.0
    assert np.isclose(regridded_da.isel(y_new=0, x_new=0).item(), expected_value)
