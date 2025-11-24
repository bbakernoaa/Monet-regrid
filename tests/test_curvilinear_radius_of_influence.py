"""Test for curvilinear interpolator with radius of influence."""

import logging

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator


def test_curvilinear_nearest_with_radius_of_influence():
    """Test that nearest neighbor interpolation works with radius of influence."""
    # Create a source grid
    source_lat = np.array([,])
    source_lon = np.array([[-1, -1],])

    source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

    # Create a target grid
    target_lat = np.array([[0.5, 1.5], [0.5, 1.5]])
    target_lon = np.array([[-0.5, -0.5], [0.5, 0.5]])

    target_grid = xr.Dataset(
        {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
    )

    # Create test data
    data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
    test_data = xr.DataArray(data_values, dims=["y", "x"])

    # Test with a radius of influence
    interpolator = CurvilinearInterpolator(
        source_grid, target_grid, method="nearest", radius_of_influence=100000
    )
    result = interpolator(test_data)

    # Result should be finite and reasonable
    assert result.shape == target_lat.shape
    assert np.all(np.isfinite(result.values))
    assert np.all(result.values >= 270.0)  # Reasonable bounds
    assert np.all(result.values <= 300.0)

    # Test with a very small radius of influence
    interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest", radius_of_influence=1)
    result = interpolator(test_data)

    # Result should be all NaNs
    assert result.shape == target_lat.shape
    assert np.all(np.isnan(result.values))


if __name__ == "__main__":
    test_curvilinear_nearest_with_radius_of_influence()
    logging.info("All tests passed!")
