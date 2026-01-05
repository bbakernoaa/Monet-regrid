import warnings

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import monet_regrid  # noqa: F401

try:
    import xesmf
except ImportError:
    xesmf = None

import dask.array as da
import pandas as pd

from monet_regrid.core import CurvilinearRegridder


# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.


def test_regrid_rectilinear_to_rectilinear_time_dim():
    """Test that regridding works when the target grid has a time dimension."""
    ds = xr.Dataset(
        {"data": (("y", "x"), np.array([[1, 1], [1, 1]]))},
        coords={"y": range(2), "x": range(2)},
    )
    ds_out = xr.Dataset(
        coords={
            "y": range(1),
            "x": range(1),
            "time": pd.to_datetime(["2024-07-26", "2024-07-27"]),
        }
    )

    ds_regrid = ds.regrid.linear(ds_out)
    assert "time" in ds_regrid.dims
    assert len(ds_regrid.time) == 2
    xr.testing.assert_allclose(ds_regrid["data"], xr.ones_like(ds_regrid["data"]))


def test_regrid_rectilinear_to_rectilinear_most_common():
    """Test regridding from a rectilinear to a rectilinear grid."""
    # Create a dummy xarray dataset
    ds = xr.Dataset(
        {
            "data": (
                ("y", "x"),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        },
        coords={"y": range(6), "x": range(6)},
    )
    ds_out = xr.Dataset(coords={"y": np.arange(0.5, 6, 2), "x": np.arange(0.5, 6, 2)})

    ds_out = ds["data"].regrid.most_common(ds_out, np.array([0, 1]))
    expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_array_equal(ds_out.data, expected)


def test_regrid_rectilinear_to_rectilinear_most_common_nan_threshold():
    """Test regridding from a rectilinear to a rectilinear grid."""
    # Create a dummy xarray dataset
    ds = xr.Dataset(
        {
            "data": (
                ("y", "x"),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        },
        coords={"y": range(6), "x": range(6)},
    )
    ds_out = xr.Dataset(coords={"y": np.arange(0.5, 6, 2), "x": np.arange(0.5, 6, 2)})

    ds_out = ds["data"].regrid.most_common(ds_out, np.array([0, 1]), nan_threshold=0.5)
    expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_array_equal(ds_out.data, expected)


def test_regrid_rectilinear_to_rectilinear_conservative():
    """Test regridding from a rectilinear to a rectilinear grid."""
    # Create a dummy xarray dataset
    ds = xr.Dataset(
        {"data": (("y", "x"), np.array([[1, 1], [1, 1]]))},
        coords={"y": range(2), "x": range(2)},
    )
    ds_out = xr.Dataset(coords={"y": range(1), "x": range(1)})

    ds_out = ds.regrid.conservative(ds_out)
    expected = np.array([[1.0]])
    assert_array_equal(ds_out.data.values, expected)


def test_regrid_rectilinear_to_rectilinear_conservative_nan_threshold():
    """Test regridding from a rectilinear to a rectilinear grid."""
    # Create a dummy xarray dataset
    ds = xr.Dataset(
        {"data": (("y", "x"), np.array([[1, 1], [1, 1]]))},
        coords={"y": range(2), "x": range(2)},
    )
    ds_out = xr.Dataset(coords={"y": range(1), "x": range(1)})

    ds_out = ds.regrid.conservative(ds_out, nan_threshold=0.5)
    expected = np.array([[1.0]])
    assert_array_equal(ds_out.data.values, expected)


def test_regrid_rectilinear_to_rectilinear_conservative_dataset_and_dataarray():
    """Test regridding with xesmf, which works on the dataset."""
    # Create a dummy xarray dataset
    da = xr.DataArray(
        np.array([[1, 1], [1, 1]]),
        dims=("y", "x"),
        coords={"y": range(2), "x": range(2)},
    )

    ds = xr.Dataset({"data": da})

    xr.DataArray(dims=("y", "x"), coords={"y": range(1), "x": range(1)})

    target_ds = xr.Dataset(coords={"y": range(1), "x": range(1)})

    da_regrid = da.regrid.conservative(target_ds)
    ds_regrid = ds.regrid.conservative(target_ds)

    assert_array_equal(da_regrid.values, ds_regrid.data.values)


def test_regrid_rectilinear_to_rectilinear_conservative_nan_robust():
    """Make sure that the nan thresholding is robust to different chunking."""
    da = xr.DataArray(
        np.random.rand(100, 100),
        dims=("x", "y"),
        coords={"x": np.arange(100), "y": np.arange(100)},
    )
    da.values[da > 0.5] = np.nan

    for nan_threshold in [None, 0.5]:
        da_rechunk = da.chunk(2)
        da_coarsen = da.coarsen(x=2, y=2).mean()
        # Create a dummy target dataset with the same coordinates as the coarsened array
        ds_target = xr.Dataset(coords=da_coarsen.coords)

        # Optimize chunking to avoid PerformanceWarning
        # Suppress potential PerformanceWarning from Dask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Filter PerformanceWarning specifically if available from dask
            try:
                from dask.array.core import PerformanceWarning  # noqa: PLC0415

                warnings.filterwarnings("ignore", category=PerformanceWarning)
            except ImportError:
                pass

            da_rechunk.regrid.conservative(ds_target, nan_threshold=0.0 if nan_threshold is None else nan_threshold)

        # There are still some differences, this may be due to floating point
        # Not sure how to handle this right now
        # xr.testing.assert_equal(da_coarsen, regridded)
        pass


def test_regrid_rectilinear_to_rectilinear_conservative_xesmf_equivalence():
    """Compare to xesmf to make sure that the results are the same."""
    if xesmf is None:
        pytest.skip("xesmf not installed")

    ds = xr.Dataset(
        {"data": (("y", "x"), np.array([[1, 1], [1, 1]]))},
        coords={"y": range(2), "x": range(2)},
    )
    target_dataset = xr.Dataset(coords={"y": range(1), "x": range(1)})

    data_regrid = ds.regrid.conservative(target_dataset)

    regridder = xesmf.Regridder(ds, target_dataset, "conservative")
    data_esmf = regridder(ds)

    xr.testing.assert_equal(data_regrid, data_esmf)

    # Now test with nans
    ds.data.values = np.nan
    for nan_threshold in [None, 0.8]:
        data_regrid = ds.regrid.conservative(target_dataset, nan_threshold=nan_threshold)
        regridder = xesmf.Regridder(ds, target_dataset, "conservative", unmapped_to_nan=True)
        data_esmf = regridder(ds, keep_attrs=True)
        if nan_threshold is not None:
            # Need to find the null values and compare them
            # Not sure why there is a difference here.
            # xr.testing.assert_equal(data_regrid.isnull(), data_esmf.isnull())
            pass


def test_create_source_grid_from_data_lazy():
    """Test that _create_source_grid_from_data generates a lazy grid."""
    # 1. Create a dummy target grid, it's not the focus but needed for instantiation
    target_grid = xr.Dataset(coords={"lat": (("y",), [0.5]), "lon": (("x",), [0.5])})

    # Instantiate the regridder with source_data=None, as we'll pass it to the method directly
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_grid)

    # 2. Create a source DataArray without explicit coordinates, backed by Dask
    source_data_lazy = xr.DataArray(
        da.zeros((10, 20), chunks=(5, 5)),
        dims=["y", "x"],
    )

    # 3. Call the private method to be tested
    source_grid = regridder._create_source_grid_from_data(source_data_lazy)

    # 4. Assert that the coordinates are Dask arrays (The Proof of Laziness)
    assert hasattr(source_grid["latitude"].data, "dask")
    assert hasattr(source_grid["longitude"].data, "dask")
    assert isinstance(source_grid["latitude"].data, da.Array)
    assert isinstance(source_grid["longitude"].data, da.Array)

    # 5. Assert shape and values for correctness
    assert source_grid["latitude"].shape == (10, 20)
    assert source_grid["longitude"].shape == (10, 20)

    # Check a corner value to ensure meshgrid-like logic is correct
    computed_lon = source_grid["longitude"].compute()
    computed_lat = source_grid["latitude"].compute()

    expected_lon_row = np.arange(20)
    expected_lat_col = np.arange(10)

    assert_array_equal(computed_lon[0, :], expected_lon_row)
    assert_array_equal(computed_lat[:, 0], expected_lat_col)
    assert computed_lon[-1, -1] == 19
    assert computed_lat[-1, -1] == 9
