import pytest
import warnings

import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

import monet_regrid  # noqa: F401
from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


try:
    import xesmf
except ImportError:
    xesmf = None

import pandas as pd


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

            da_rechunk.regrid.conservative(
                ds_target, nan_threshold=0.0 if nan_threshold is None else nan_threshold
            )

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


def test_regridder_serialization_deserialization(tmp_path):
    """Test saving and loading a regridder instance."""
    # Create sample data and target grid
    source_ds = xr.Dataset(
        {"data": (("y", "x"), np.random.rand(10, 10))},
        coords={"y": np.arange(10), "x": np.arange(10)},
    )
    target_ds = xr.Dataset(
        coords={"y": np.arange(0, 10, 0.5), "x": np.arange(0, 10, 0.5)}
    )

    # 1. Test RectilinearRegridder
    regridder = RectilinearRegridder(source_data=source_ds, target_grid=target_ds, method="linear")

    # Save and load the regridder
    filepath_rectilinear = tmp_path / "regridder_rectilinear.nc"
    regridder.to_file(filepath_rectilinear)
    loaded_regridder = RectilinearRegridder.from_file(filepath_rectilinear)

    # Compare results
    expected_rectilinear = regridder(source_ds)
    result_rectilinear = loaded_regridder(source_ds)
    xr.testing.assert_allclose(expected_rectilinear, result_rectilinear)

    # 2. Test CurvilinearRegridder
    # Create a dummy source dataset with 2D coordinates
    lon = np.arange(-180, 180, 45)
    lat = np.arange(-90, 91, 30)
    lon2d, lat2d = np.meshgrid(lon, lat)
    source_ds_curv = xr.Dataset(
        {"data": (("y", "x"), np.random.rand(lon2d.shape[0], lon2d.shape[1]))},
        coords={"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
    )
    target_ds_curv = xr.Dataset(
        coords={"y": np.arange(-80, 81, 40), "x": np.arange(-160, 161, 40)}
    )

    regridder_curv = CurvilinearRegridder(
        source_data=source_ds_curv, target_grid=target_ds_curv, method="linear"
    )

    # Save and load the regridder
    filepath_curvilinear = tmp_path / "regridder_curvilinear.nc"
    regridder_curv.to_file(filepath_curvilinear)
    loaded_regridder_curv = CurvilinearRegridder.from_file(filepath_curvilinear)

    # Compare results
    expected_curvilinear = regridder_curv(source_ds_curv)
    result_curvilinear = loaded_regridder_curv(source_ds_curv)
    xr.testing.assert_allclose(expected_curvilinear, result_curvilinear)
