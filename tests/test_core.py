import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def test_create_source_grid_from_data_lazy_generation():
    """
    Test lazy coordinate generation in `_create_source_grid_from_data`.

    This test verifies that when a Dask-backed DataArray is passed without
    explicit coordinates, the method correctly generates lazy, broadcasted
    latitude and longitude coordinates as Dask arrays.
    """
    # 1. Create a Dask-backed DataArray without explicit coordinates
    data_values = da.random.random((10, 20), chunks=(5, 10))
    source_da = xr.DataArray(data_values, dims=["y", "x"])

    # A minimal target grid is needed for the regridder's constructor,
    # though it is not used in this specific method call.
    target_ds = xr.Dataset(
        coords={
            "latitude": (("y_new",), np.arange(2)),
            "longitude": (("x_new",), np.arange(2)),
        }
    )

    # 2. Instantiate a mock regridder to test the internal method
    # We pass source_data=None because we are testing the method directly
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_ds)

    # 3. Call the internal method to generate the source grid
    source_grid = regridder._create_source_grid_from_data(source_da)

    # 4. Assert that the generated coordinates are Dask arrays
    assert "latitude" in source_grid.coords
    assert "longitude" in source_grid.coords
    assert isinstance(source_grid["latitude"].data, da.Array)
    assert isinstance(source_grid["longitude"].data, da.Array)

    # 5. Assert correct shape and dimension names
    assert source_grid["latitude"].shape == (10, 20)
    assert source_grid["longitude"].shape == (10, 20)
    assert source_grid["latitude"].dims == ("y", "x")
    assert source_grid["longitude"].dims == ("y", "x")

    # 6. Verify the computed values to ensure linspace and broadcasting are correct
    # The key change is ensuring the test matches xarray's broadcasting behavior,
    # which is different from np.meshgrid(..., indexing='ij').
    computed_lat = source_grid["latitude"].compute()
    computed_lon = source_grid["longitude"].compute()

    y_coords = np.linspace(0, 9, 10)
    x_coords = np.linspace(0, 19, 20)

    # Replicate xarray.broadcast's behavior for the test assertion
    expected_lat_2d = np.broadcast_to(y_coords[:, np.newaxis], (10, 20))
    expected_lon_2d = np.broadcast_to(x_coords, (10, 20))

    np.testing.assert_allclose(computed_lat, expected_lat_2d)
    np.testing.assert_allclose(computed_lon, expected_lon_2d)


def test_create_source_grid_from_data_with_explicit_coords():
    """
    Test that `_create_source_grid_from_data` uses existing coordinates.

    This test ensures that if the input DataArray already has CF-compliant
    latitude and longitude coordinates, the method correctly extracts them
    instead of generating new ones.
    """
    # 1. Create a DataArray with explicit lat/lon coordinates
    lat = np.random.rand(10, 20)
    lon = np.random.rand(10, 20)
    source_da = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], lat), "longitude": (["y", "x"], lon)},
    )

    target_ds = xr.Dataset(coords={"latitude": (("y_new",), [1]), "longitude": (("x_new",), [1])})
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_ds)

    # 2. Call the method
    source_grid = regridder._create_source_grid_from_data(source_da)

    # 3. Assert that the returned grid contains the original coordinates
    assert "latitude" in source_grid.coords
    assert "longitude" in source_grid.coords
    xr.testing.assert_equal(source_grid["latitude"], source_da["latitude"])
    xr.testing.assert_equal(source_grid["longitude"], source_da["longitude"])


def test_create_source_grid_from_data_insufficient_dims():
    """
    Test that a ValueError is raised for data with fewer than 2 dimensions.
    """
    source_da = xr.DataArray(np.random.rand(10), dims=["x"])
    target_ds = xr.Dataset(coords={"latitude": (("y_new",), [1]), "longitude": (("x_new",), [1])})
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_ds)

    with pytest.raises(ValueError, match="Source data must have at least 2 dimensions"):
        regridder._create_source_grid_from_data(source_da)


def test_curvilinear_regridder_initialization():
    """Test the initialization of the CurvilinearRegridder.
    This test verifies that the `method` and `method_kwargs` attributes
    are correctly assigned during the instantiation of the regridder.
    """
    # 1. Create dummy source and target grids for initialization
    source_da = xr.DataArray(
        np.random.rand(2, 3),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], np.random.rand(2, 3)), "longitude": (["y", "x"], np.random.rand(2, 3))},
    )
    target_ds = xr.Dataset(coords={"latitude": (("y_new",), [1.0]), "longitude": (("x_new",), [1.0])})

    # 2. Test initialization with a specific method and kwargs
    regridder_custom = CurvilinearRegridder(
        source_data=source_da,
        target_grid=target_ds,
        method="nearest",
        k=3,
        radius_of_influence=50000,
    )
    assert regridder_custom.method == "nearest"
    expected_kwargs = {"k": 3, "radius_of_influence": 50000}
    assert regridder_custom.method_kwargs == expected_kwargs

    # 3. Test initialization with default parameters
    regridder_default = CurvilinearRegridder(
        source_data=source_da,
        target_grid=target_ds,
    )
    assert regridder_default.method == "linear"
    assert regridder_default.method_kwargs == {}
