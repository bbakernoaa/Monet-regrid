from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


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


def test_curvilinear_regridder_coordinate_identification():
    """Test that CurvilinearRegridder identifies coordinate names once."""
    # Create source and target grids
    source_x, source_y = np.meshgrid(np.arange(5), np.arange(5))
    source_lat = 30 + 0.1 * source_y
    source_lon = -100 + 0.1 * source_x
    xr.Dataset(
        {"lat": (["y", "x"], source_lat), "lon": (["y", "x"], source_lon)},
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
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

    # Use the accessor to create the regridder
    regridder = data.regrid.build_regridder(ds_target_grid=target_grid, method="linear")

    # Check that the regridder has correctly identified the coordinate names
    assert regridder.source_lat_name == "lat"
    assert regridder.source_lon_name == "lon"
    assert regridder.target_lat_name == "latitude"
    assert regridder.target_lon_name == "longitude"


def test_curvilinear_regridder_caches_interpolator():
    """Test that CurvilinearRegridder caches the interpolator object."""
    # Create source and target grids
    source_da = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        coords={
            "lat": (("y", "x"), np.random.rand(5, 5) * 90),
            "lon": (("y", "x"), np.random.rand(5, 5) * 360),
        },
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y_new", "x_new"), np.random.rand(3, 3) * 90),
            "lon": (("y_new", "x_new"), np.random.rand(3, 3) * 360),
        }
    )

    regridder = CurvilinearRegridder(source_da, target_ds, method="linear")

    # Use patch to spy on the CurvilinearInterpolator constructor
    with patch("monet_regrid.core.CurvilinearInterpolator", autospec=True) as mock_interpolator:
        # First call - should create and cache an interpolator
        regridder()
        mock_interpolator.assert_called_once()

        # Second call with same parameters - should use the cached interpolator
        regridder()
        mock_interpolator.assert_called_once()  # Still called only once

        # Third call with different method - should create a new interpolator
        regridder(method="nearest")
        assert mock_interpolator.call_count == 2

        # Fourth call with the original method - should use the cache again
        regridder()
        assert mock_interpolator.call_count == 2


def test_curvilinear_interpolator_with_1d_coords():
    """Test interpolation with 1D source and target coordinates."""
    # Create source grid with 1D coordinates
    source_lat = np.arange(30, 35)
    source_lon = np.arange(-100, -95)
    xr.Dataset(coords={"latitude": source_lat, "longitude": source_lon})

    # Create target grid with 1D coordinates
    target_lat = np.linspace(30.5, 33.5, 3)
    target_lon = np.linspace(-99.5, -96.5, 4)
    target_grid = xr.Dataset(coords={"latitude": target_lat, "longitude": target_lon})

    # Create test data
    data_values = np.random.rand(len(source_lat), len(source_lon))
    test_data = xr.DataArray(
        data_values,
        dims=["latitude", "longitude"],
        coords={"latitude": source_lat, "longitude": source_lon},
    )

    # Test interpolation
    result = test_data.regrid.nearest(target_grid)

    assert result.shape == (len(target_lat), len(target_lon))


def test_curvilinear_interpolator_invalid_input_type():
    """Test that a TypeError is raised for invalid input types."""
    # Create dummy grids
    source_grid = xr.Dataset({"lat": (("y", "x"), np.zeros((2, 2))), "lon": (("y", "x"), np.zeros((2, 2)))})
    target_grid = xr.Dataset({"lat": (("y_t", "x_t"), np.ones((3, 3))), "lon": (("y_t", "x_t"), np.ones((3, 3)))})
    regridder = CurvilinearRegridder(source_grid, target_grid)

    with pytest.raises(ValueError):
        regridder(xr.DataArray(np.zeros((2, 2)), dims=["y", "x"]))


def test_curvilinear_interpolator_data_validation_error():
    """Test that a ValueError is raised for mismatched data coordinates."""
    # Create source and target grids
    source_da = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        coords={
            "lat": (("y", "x"), np.random.rand(5, 5) * 90),
            "lon": (("y", "x"), np.random.rand(5, 5) * 360),
        },
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y_new", "x_new"), np.random.rand(3, 3) * 90),
            "lon": (("y_new", "x_new"), np.random.rand(3, 3) * 360),
        }
    )

    regridder = CurvilinearRegridder(source_da, target_ds)

    # Create data with incorrect dimensions
    mismatched_data = xr.DataArray(np.random.rand(4, 4), dims=["y", "x"])

    with pytest.raises(ValueError, match="Could not identify latitude coordinate"):
        regridder(mismatched_data)


def test_curvilinear_attribute_errors():
    """Test that attribute errors are raised for inappropriate methods."""
    # Create dummy grids
    source_da = xr.DataArray(
        np.random.rand(2, 2),
        dims=["y", "x"],
        coords={
            "lat": (("y", "x"), np.zeros((2, 2))),
            "lon": (("y", "x"), np.zeros((2, 2))),
        },
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y_t", "x_t"), np.ones((3, 3))),
            "lon": (("y_t", "x_t"), np.ones((3, 3))),
        }
    )

    # Use 'nearest' method, which doesn't create triangles
    regridder = CurvilinearRegridder(source_da, target_ds, method="nearest")
    interpolator = regridder(source_da)

    with pytest.raises(AttributeError):
        interpolator.triangles

    with pytest.raises(AttributeError):
        interpolator.convex_hull
