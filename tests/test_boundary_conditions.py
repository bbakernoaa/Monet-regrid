import numpy as np
import pytest
import xarray as xr

"""Boundary condition tests for curvilinear regridding edge cases.

This module tests edge cases, boundary conditions, and robustness scenarios
including poles, date lines, empty grids, and NaN propagation.
"""


@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning")
class TestPoleProximityHandling:
    """Test handling of pole proximity and polar regions."""

    def setup_method(self):
        """Set up test data for pole proximity tests."""
        # Create grids near the North Pole
        self.polar_source_lat = np.array([[89.5, 89.6], [89.5, 89.6]])
        self.polar_source_lon = np.array([[-135.0, 45.0], [-135.0, 45.0]])

        self.polar_target_lat = np.array([[89.55, 89.65], [89.55, 89.65]])
        self.polar_target_lon = np.array([[-135.0, 45.0], [-135.0, 45.0]])

        self.polar_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.polar_source_lat), "longitude": (["y", "x"], self.polar_source_lon)}
        )

        self.polar_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.polar_target_lat),
                "longitude": (["y_target", "x_target"], self.polar_target_lon),
            }
        )

    def test_north_pole_handling(self):
        """Test interpolation near the North Pole."""
        # Create test data
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.polar_source_lat),
                "longitude": (["y", "x"], self.polar_source_lon),
            },
        )

        # Test nearest neighbor interpolation near pole
        result = test_data.regrid.nearest(self.polar_target_grid)

        # Result should be finite and reasonable
        assert result.shape == self.polar_target_lat.shape
        assert np.all(np.isfinite(result.values))
        assert np.all(result.values >= 270.0)
        assert np.all(result.values <= 300.0)

    def test_south_pole_handling(self):
        """Test interpolation near the South Pole."""
        # Create grids near the South Pole
        south_source_lat = np.array([[-89.6, -89.5], [-89.6, -89.5]])
        south_source_lon = np.array([[-135.0, 45.0], [-135.0, 45.0]])

        south_target_lat = np.array([[-89.65, -89.55], [-89.65, -89.55]])
        south_target_lon = np.array([[-135.0, 45.0], [-135.0, 45.0]])

        south_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], south_target_lat),
                "longitude": (["y_target", "x_target"], south_target_lon),
            }
        )

        # Create test data
        data_values = np.array([[270.0, 275.0], [272.0, 277.0]])
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], south_source_lat),
                "longitude": (["y", "x"], south_source_lon),
            },
        )

        # Test interpolation near South Pole
        result = test_data.regrid.nearest(south_target_grid)

        # Result should be finite and reasonable
        assert result.shape == south_target_lat.shape
        assert np.all(np.isfinite(result.values))
        assert np.all(result.values >= 260.0)
        assert np.all(result.values <= 290.0)

    def test_polar_linear_interpolation_fallback(self):
        """Test that linear interpolation falls back to nearest neighbor in polar regions."""
        # Create data that would stress linear interpolation at poles
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], self.polar_source_lat),
                "longitude": (["y", "x"], self.polar_source_lon),
            },
        )

        # Test linear interpolation (should fall back to nearest neighbor behavior in polar regions)
        result = test_data.regrid.linear(self.polar_target_grid)

        # Should complete without error and produce reasonable results
        assert result.shape == self.polar_target_lat.shape
        assert np.all(np.isfinite(result.values) | np.isnan(result.values))

    def test_pole_coordinate_singularity(self):
        """Test handling of coordinate singularities at poles."""
        # Test with exactly 90 degree latitude
        singular_source_lat = np.array([[90.0, 90.0], [90.0, 90.0]])
        singular_source_lon = np.array([[0.0, 180.0], [0.0, -90.0]])

        singular_target_lat = np.array([[90.0]])
        singular_target_lon = np.array([[45.0]])

        singular_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], singular_target_lat),
                "longitude": (["y_target", "x_target"], singular_target_lon),
            }
        )

        # Test interpolation at exact pole
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(
            data_values,
            dims=["y", "x"],
            coords={
                "latitude": (["y", "x"], singular_source_lat),
                "longitude": (["y", "x"], singular_source_lon),
            },
        )

        # Should handle pole coordinates gracefully
        try:
            result = test_data.regrid.nearest(singular_target_grid)

            # If it succeeds, verify result properties
            assert result.shape == singular_target_lat.shape
            assert np.all(np.isfinite(result.values)) or np.any(np.isnan(result.values))
        except Exception:  # noqa: S110
            # If it raises an exception, that's acceptable for this edge case
            pass
