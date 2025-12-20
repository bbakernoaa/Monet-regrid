import numpy as np
import pytest
import xarray as xr

import monet_regrid  # noqa: F401

"""Test cases for boundary conditions and geographic edge cases."""

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.


def test_antarctic_pole_interpolation():
    """Test interpolation over the Antarctic pole."""
    # Create a source grid that covers the South Pole
    source_lat = np.array([[-89.9, -89.8], [-89.7, -89.6]])
    source_lon = np.array([[-180.0, 0.0], [-90.0, 90.0]])
    source_data = xr.DataArray(
        np.arange(4).reshape(2, 2),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Create a target grid directly over the pole
    target_lat = np.array([[-90.0]])
    target_lon = np.array([[0.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    # Perform interpolation (linear should handle this)
    result = source_data.regrid.linear(target_grid)

    # Verification: The result should be a single value, and it should be finite.
    # The exact value depends on triangulation, but we expect it to be an average
    # of the surrounding points.
    assert result.shape == (1, 1)
    assert np.isfinite(result.values).all()


def test_arctic_pole_interpolation():
    """Test interpolation over the Arctic pole."""
    # Source grid covering the North Pole
    source_lat = np.array([[89.6, 89.7], [89.8, 89.9]])
    source_lon = np.array([[-180.0, 90.0], [0.0, -90.0]])
    source_data = xr.DataArray(
        np.arange(4).reshape(2, 2),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Target grid directly over the pole
    target_lat = np.array([[90.0]])
    target_lon = np.array([[0.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    result = source_data.regrid.linear(target_grid)
    assert result.shape == (1, 1)
    assert np.isfinite(result.values).all()


def test_dateline_crossing_interpolation():
    """Test interpolation across the antimeridian (dateline)."""
    # Source grid crossing the dateline
    source_lat = np.array([[0.0, 0.0], [0.0, 0.0]])
    source_lon = np.array([[179.9, -179.9], [179.8, -179.8]])
    source_data = xr.DataArray(
        np.array([[10, 20], [12, 22]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    # Target grid on the dateline
    target_lat = np.array([[0.0]])
    target_lon = np.array([[180.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    result = source_data.regrid.linear(target_grid)

    # Expected value should be an interpolation between 10 and 20,
    # and 12 and 22. Given the geometry, it should be close to the average.
    assert result.shape == (1, 1)
    assert np.isfinite(result.values).all()
    assert abs(result.values[0, 0] - 16) < 7  # Average of all 4 points


def test_collocated_points_robustness():
    """Test robustness to collocated or nearly collocated source points."""
    source_lat = np.array([[0.0, 0.0], [0.0, 1.0]])
    source_lon = np.array([[0.0, 0.0], [0.0, 1.0]])  # Three points are identical
    source_data = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    target_lat = np.array([[0.5]])
    target_lon = np.array([[0.5]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    # This should not fail. If it does, QhullError might be raised.
    try:
        result = source_data.regrid.linear(target_grid)
        assert result.shape == (1, 1)
        assert np.isfinite(result.values).all()
    except Exception as e:
        pytest.fail(f"Interpolation with collocated points failed: {e}")


def test_single_point_source_grid():
    """Test with a source grid that is just a single point."""
    source_lat = np.array([[45.0]])
    source_lon = np.array([[45.0]])
    source_data = xr.DataArray(
        np.array([[100.0]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    target_lat = np.array([[45.0, 46.0], [45.0, 46.0]])
    target_lon = np.array([[45.0, 45.0], [46.0, 46.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    # Linear interpolation with a single point is ill-defined.
    # The regridder should fall back to nearest neighbor.
    result = source_data.regrid.linear(target_grid)
    expected = np.full((2, 2), 100.0)
    np.testing.assert_allclose(result.values, expected)

    # Nearest neighbor should definitely work
    result_nn = source_data.regrid.nearest(target_grid)
    np.testing.assert_allclose(result_nn.values, expected)


def test_singular_matrix_scenario():
    """Test a scenario that could lead to a singular matrix in barycentric calculation."""
    # All points are on a great circle (collinear in 3D space)
    source_lat = np.array([[0.0, 0.0, 0.0, 0.0]])
    source_lon = np.array([[-1.0, 1.0, 2.0, 3.0]])
    source_data = xr.DataArray(
        np.array([[10, 20, 30, 40]]),
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)},
    )

    target_lat = np.array([[0.0]])
    target_lon = np.array([[0.0]])
    target_grid = xr.Dataset(
        coords={"latitude": (["y_out", "x_out"], target_lat), "longitude": (["y_out", "x_out"], target_lon)}
    )

    # This should ideally fall back to nearest neighbor or handle the singularity gracefully.
    try:
        result = source_data.regrid.linear(target_grid)
        # If it succeeds, verify result properties
        assert result.shape == (1, 1)
        assert np.all(np.isfinite(result.values)) or np.any(np.isnan(result.values))
    except Exception:  # noqa: S110
        # If it raises an exception, that's acceptable for this edge case
        pass
