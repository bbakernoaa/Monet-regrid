from unittest.mock import patch

import numpy as np
import xarray as xr

import monet_regrid


def test_curvilinear_interpolator_nearest_interpolation():
    """Test nearest neighbor interpolation."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

    target_x, target_y = np.meshgrid(np.linspace(0, 4, 3), np.linspace(0, 5, 4))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.random.rand(6, 5)  # (y, x)
    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Test nearest neighbor interpolation
    result = test_data.regrid.nearest(target_grid)

    # Check result dimensions
    assert result.shape == target_lat.shape
    assert "y_target" in result.dims
    assert "x_target" in result.dims


def test_curvilinear_interpolator_nearest_interpolation_with_time():
    """Test nearest neighbor interpolation with additional dimensions."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0, 2, 2), np.linspace(0, 3, 3))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data with time dimension
    time_dim = 5
    data_values = np.random.rand(time_dim, 4, 3)  # (time, y, x)
    test_data = xr.DataArray(
        data_values,
        dims=["time", "y", "x"],
        coords={
            "time": range(time_dim),
            "latitude": (["y", "x"], source_lat),
            "longitude": (["y", "x"], source_lon),
        },
    )

    # Test nearest neighbor interpolation
    result = test_data.regrid.nearest(target_grid)

    # Check result dimensions - should have time and target grid dimensions
    expected_shape = (time_dim, 3, 2)  # (time, y_target, x_target)
    assert result.shape == expected_shape
    assert "time" in result.dims
    assert "y_target" in result.dims
    assert "x_target" in result.dims


def test_curvilinear_interpolator_dataset_interpolation():
    """Test interpolation of entire datasets."""
    # Create simple curvilinear grids
    source_x, source_y = np.meshgrid(np.arange(3), np.arange(3))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0, 2, 2), np.linspace(0, 2, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test dataset
    data_values = np.random.rand(3, 3)
    test_dataset = xr.Dataset(
        {
            "var1": (["y", "x"], data_values),
            "var2": (["y", "x"], data_values * 2),
            "other_var": (("time",), np.arange(1)),  # This should be preserved as-is
        },
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Test dataset interpolation
    result = test_dataset.regrid.nearest(target_grid)

    # Check that interpolated variables have correct shape
    assert result["var1"].shape == (2, 2)
    assert result["var2"].shape == (2, 2)
    # Check that non-spatial variable is preserved
    assert "other_var" in result
    np.testing.assert_array_equal(result["other_var"].values, test_dataset["other_var"].values)

    # Check that target coordinates are added
    assert "y_target" in result.coords
    assert "x_target" in result.coords


def test_curvilinear_interpolator_linear_interpolation():
    """Test linear interpolation (basic functionality)."""
    source_x, source_y = np.meshgrid(np.arange(4), np.arange(4))
    source_lat = 30 + 0.5 * source_x + 0.1 * source_y + 0.0001 * np.random.rand(*source_x.shape)
    source_lon = -100 + 0.3 * source_x + 0.2 * source_y

    target_x, target_y = np.meshgrid(np.linspace(0.5, 2.5, 2), np.linspace(0.5, 2.5, 2))
    target_lat = 30 + 0.5 * target_x + 0.1 * target_y
    target_lon = -100 + 0.3 * target_x + 0.2 * target_y

    xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.ones((4, 4)) * 5.0  # Simple constant data
    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Test linear interpolation
    result = test_data.regrid.linear(target_grid)

    # With constant data, result should be approximately the same value
    assert result.shape == target_lat.shape
    # Values should be close to 5.0 (the original constant value)
    np.testing.assert_allclose(result.data, 5.0, rtol=1e-5)


@patch("monet_regrid.core.CurvilinearInterpolator")
def test_curvilinear_regridder_coordinate_identification(mock_interpolator):
    """Test that CurvilinearRegridder identifies and passes coordinate names correctly."""
    # Create source and target grids
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(5))
    source_lat = 30 + 0.1 * source_y
    source_lon = -100 + 0.1 * source_x
    data = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        coords={"lat": (["y", "x"], source_lat), "lon": (["y", "x"], source_lon)},
    )

    target_x, target_y = np.meshgrid(np.arange(3), np.arange(3))
    target_lat = 30 + 0.1 * target_y
    target_lon = -100 + 0.1 * target_x
    target_grid = xr.Dataset(
        {"latitude": (["y_t", "x_t"], target_lat), "longitude": (["y_t", "x_t"], target_lon)},
        coords={"x_t": np.arange(3), "y_t": np.arange(3)},
    )

    # Use the accessor to perform regridding
    data.regrid.linear(target_grid)

    # Check that the interpolator was called with the correct coordinate names
    mock_interpolator.assert_called_once()
    _, call_kwargs = mock_interpolator.call_args
    assert call_kwargs.get("source_lat_name") == "lat"
    assert call_kwargs.get("source_lon_name") == "lon"
    assert call_kwargs.get("target_lat_name") == "latitude"
    assert call_kwargs.get("target_lon_name") == "longitude"


def test_rectilinear_to_curvilinear_regridding():
    """Test regridding from a rectilinear source to a curvilinear target."""
    # 1. Source Grid (Rectilinear)
    source_ds = xr.Dataset(
        {
            "foo": (("lat", "lon"), np.random.rand(5, 10)),
        },
        coords={
            "lat": np.linspace(30, 40, 5),
            "lon": np.linspace(-120, -110, 10),
        },
    )

    # 2. Target Grid (Curvilinear)
    target_x, target_y = np.meshgrid(np.arange(3), np.arange(4))
    target_lat = 32 + 0.5 * target_y
    target_lon = -118 + 0.5 * target_x
    target_grid = xr.Dataset(
        {"latitude": (("y", "x"), target_lat), "longitude": (("y", "x"), target_lon)},
    )

    # 3. Perform regridding
    result = source_ds["foo"].regrid.linear(target_grid)

    # 4. Verification
    # Check that the result has the expected shape of the target grid
    assert result.shape == target_lat.shape

    # Check that the coordinates of the result match the target grid
    assert "latitude" in result.coords
    assert "longitude" in result.coords
    np.testing.assert_allclose(result.coords["latitude"], target_lat)
    np.testing.assert_allclose(result.coords["longitude"], target_lon)
