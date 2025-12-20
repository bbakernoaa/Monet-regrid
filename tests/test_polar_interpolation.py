"""Test interpolation at and around the poles."""

import numpy as np
import xarray as xr


def test_arctic_pole_interpolation_nearest():
    """Test nearest neighbor interpolation over the Arctic pole."""
    # Source grid covering the North Pole
    source_lat = np.array([[89.8, 89.9], [89.8, 89.9]])
    source_lon = np.array([[-180.0, 0.0], [90.0, -90.0]])
    source_data = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Target grid directly at the pole
    target_lat = np.array([[90.0]])
    target_lon = np.array([[0.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    result = source_data.regrid.nearest(target_grid)

    # The point at (89.9, 0.0) is the closest. Its value is 2.
    assert result.item() == 2


def test_antarctic_pole_interpolation_linear():
    """Test linear interpolation over the Antarctic pole."""
    # Source grid surrounding the South Pole
    source_lat = np.array([[-89.8, -89.8], [-89.9, -89.9]])
    source_lon = np.array([[-90.0, 90.0], [-90.0, 90.0]])
    source_data = xr.DataArray(
        np.array([[10, 20], [15, 25]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Target grid directly at the pole
    target_lat = np.array([[-90.0]])
    target_lon = np.array([[0.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    result = source_data.regrid.linear(target_grid)

    # With the pole as the target, the result should be the average
    # of the equidistant points at the highest latitude.
    # In 3D space, the pole is equidistant from all points on a circle of latitude.
    # So, the interpolation should be the average of the values at -89.9 lat.
    expected_value = 15
    assert np.isclose(result.item(), expected_value)
