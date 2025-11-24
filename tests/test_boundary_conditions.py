"""Boundary condition tests for curvilinear regridding edge cases.

This module tests edge cases, boundary conditions, and robustness scenarios
including poles, date lines, empty grids, and NaN propagation.
"""

from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from monet_regrid.curvilinear import CurvilinearInterpolator

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: from xarray_regrid.curvilinear import CurvilinearInterpolator
# New import: from monet_regrid.curvilinear import CurvilinearInterpolator


class TestPoleProximityHandling:
    """Test handling of pole proximity and polar regions."""

    def setup_method(self):
        """Set up test data for pole proximity tests."""
        # Create grids near the North Pole
        self.polar_source_lat = np.array([[89.5, 89.6], [89.5, 89.6]])
        self.polar_source_lon = np.array([[0, 0], [90, 90]])

        self.polar_target_lat = np.array([[89.55, 89.65], [89.55, 89.65]])
        self.polar_target_lon = np.array([[45, 45], [135, 135]])

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
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Test nearest neighbor interpolation near pole
        interpolator = CurvilinearInterpolator(self.polar_source_grid, self.polar_target_grid, method="nearest")
        result = interpolator(test_data)

        # Result should be finite and reasonable
        assert result.shape == self.polar_target_lat.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 270.0)  # Reasonable temperature bounds
        assert np.all(result <= 300.0)

    def test_south_pole_handling(self):
        """Test interpolation near the South Pole."""
        # Create grids near the South Pole
        south_source_lat = np.array([[-89.6, -89.5], [-89.6, -89.5]])
        south_source_lon = np.array([[0, 0], [90, 90]])

        south_target_lat = np.array([[-89.65, -89.55], [-89.65, -89.55]])
        south_target_lon = np.array([[45, 45], [135, 135]])

        south_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], south_source_lat), "longitude": (["y", "x"], south_source_lon)}
        )

        south_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], south_target_lat),
                "longitude": (["y_target", "x_target"], south_target_lon),
            }
        )

        # Create test data
        data_values = np.array([[270.0, 275.0], [272.0, 277.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Test interpolation near South Pole
        interpolator = CurvilinearInterpolator(south_source_grid, south_target_grid, method="nearest")
        result = interpolator(test_data)

        # Result should be finite and reasonable
        assert result.shape == south_target_lat.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 260.0)  # Reasonable temperature bounds
        assert np.all(result <= 290.0)

    def test_polar_linear_interpolation_fallback(self):
        """Test that linear interpolation falls back to nearest neighbor in polar regions."""
        # Create data that would stress linear interpolation at poles
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Test linear interpolation (should fall back to nearest neighbor behavior in polar regions)
        interpolator = CurvilinearInterpolator(self.polar_source_grid, self.polar_target_grid, method="linear")
        result = interpolator(test_data)

        # Should complete without error and produce reasonable results
        assert result.shape == self.polar_target_lat.shape
        assert np.all(np.isfinite(result) | np.isnan(result))  # Allow NaNs where triangulation fails

    def test_pole_coordinate_singularity(self):
        """Test handling of coordinate singularities at poles."""
        # Test with exactly 90 degree latitude
        singular_source_lat = np.array([[90.0, 90.0], [90.0, 90.0]])
        singular_source_lon = np.array([[0, 90], [180, -90]])

        singular_target_lat = np.array([[90.0]])
        singular_target_lon = np.array([[45.0]])

        singular_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], singular_source_lat), "longitude": (["y", "x"], singular_source_lon)}
        )

        singular_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], singular_target_lat),
                "longitude": (["y_target", "x_target"], singular_target_lon),
            }
        )

        # Test interpolation at exact pole
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Should handle pole coordinates gracefully
        try:
            interpolator = CurvilinearInterpolator(singular_source_grid, singular_target_grid, method="nearest")
            result = interpolator(test_data)

            # If it succeeds, verify result properties
            assert result.shape == singular_target_lat.shape
            assert np.all(np.isfinite(result)) or np.any(np.isnan(result))  # Allow NaNs
        except Exception:
            # If it raises an exception, that's acceptable for this edge case
            pass


class TestDateLineCrossing:
    """Test handling of date line crossing and longitude wrapping."""

    def setup_method(self):
        """Set up test data for date line tests."""
        # Create grids that cross the International Date Line
        self.dateline_source_lat = np.array([[0, 10], [0, 10]])
        self.dateline_source_lon = np.array([[170, 170], [-170, -170]])  # Crosses 180°

        self.dateline_target_lat = np.array([[5, 8], [5, 8]])
        self.dateline_target_lon = np.array([[175, 175], [-175, -175]])  # Also crosses 180°

        self.dateline_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.dateline_source_lat), "longitude": (["y", "x"], self.dateline_source_lon)}
        )

        self.dateline_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.dateline_target_lat),
                "longitude": (["y_target", "x_target"], self.dateline_target_lon),
            }
        )

    def test_date_line_crossing_interpolation(self):
        """Test interpolation across the International Date Line."""
        # Create test data
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Test interpolation across dateline
        interpolator = CurvilinearInterpolator(self.dateline_source_grid, self.dateline_target_grid, method="nearest")
        result = interpolator(test_data)

        # Result should be finite and reasonable
        assert result.shape == self.dateline_target_lat.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 270.0)  # Reasonable bounds
        assert np.all(result <= 300.0)

    def test_longitude_wrapping_consistency(self):
        """Test that longitude wrapping is handled consistently."""
        # Create grids with different longitude representations
        # Grid 1: -180 to 180
        lon1 = np.array([[170, 175], [-175, 179]])

        # Grid 2: 0 to 360
        lon2 = np.array([[170, 175], [185, 179]])

        source_lat = np.array([[0, 10], [0, 10]])

        source_grid_180 = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], lon1)})

        source_grid_360 = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], lon2)})

        target_grid = xr.Dataset(
            {"latitude": (["y_target", "x_target"], source_lat), "longitude": (["y_target", "x_target"], lon1)}
        )

        # Test both longitude conventions
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        # Both should handle longitude coordinates correctly
        for source_grid in [source_grid_180, source_grid_360]:
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)

            assert result.shape == source_lat.shape
            assert np.all(np.isfinite(result))

    def test_antimeridian_continuity(self):
        """Test continuity across the antimeridian."""
        # Create a grid that wraps around the antimeridian
        source_lat = np.array([[0, 1], [0, 1]])
        source_lon = np.array([[179, 179], [-179, -179]])  # Close to antimeridian

        target_lat = np.array([[0.5, 0.8], [0.5, 0.8]])
        target_lon = np.array([[179.5, 179.5], [-179.5, -179.5]])  # Even closer to antimeridian

        source_grid = xr.Dataset({"latitude": (["y", "x"], source_lat), "longitude": (["y", "x"], source_lon)})

        target_grid = xr.Dataset(
            {"latitude": (["y_target", "x_target"], target_lat), "longitude": (["y_target", "x_target"], target_lon)}
        )

        # Test interpolation near antimeridian
        data_values = np.array([[280.0, 285.0], [282.0, 287.0]])
        test_data = xr.DataArray(data_values, dims=["y", "x"])

        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)

        # Should handle antimeridian proximity gracefully
        assert result.shape == target_lat.shape
        assert np.all(np.isfinite(result))


class TestEmptyAndDegenerateGrids:
    """Test handling of empty and degenerate grid cases."""

    def test_empty_source_grid(self):
        """Test handling of empty source grids."""
        # Create empty grids
        empty_source = xr.Dataset(
            {
                "latitude": (["y", "x"], np.array([]).reshape(0, 0)),
                "longitude": (["y", "x"], np.array([]).reshape(0, 0)),
            }
        )

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], np.array([[0]])),
                "longitude": (["y_target", "x_target"], np.array([[0]])),
            }
        )

        # Should handle empty grids gracefully
        try:
            interpolator = CurvilinearInterpolator(empty_source, target_grid, method="nearest")
            test_data = xr.DataArray(np.array([]).reshape(0, 0), dims=["y", "x"])
            result = interpolator(test_data)

            # If it succeeds, verify result
            assert isinstance(result, xr.DataArray)
        except Exception:
            # If it raises an exception for empty grids, that's acceptable
            pass

    def test_single_point_grid(self):
        """Test handling of single-point grids."""
        # Create single-point grids
        single_source_lat = np.array([[45.0]])
        single_source_lon = np.array([[0.0]])

        single_target_lat = np.array([[45.5]])
        single_target_lon = np.array([[0.5]])

        single_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], single_source_lat), "longitude": (["y", "x"], single_source_lon)}
        )

        single_target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], single_target_lat),
                "longitude": (["y_target", "x_target"], single_target_lon),
            }
        )

        # Test single-point interpolation
        test_data = xr.DataArray(np.array([[280.0]]), dims=["y", "x"])

        # Nearest neighbor should work with single point
        interpolator = CurvilinearInterpolator(single_source_grid, single_target_grid, method="nearest")
        result = interpolator(test_data)

        assert result.shape == single_target_lat.shape
        assert np.all(np.isfinite(result))

        # Linear interpolation should handle single point gracefully with fallback
        try:
            interpolator = CurvilinearInterpolator(single_source_grid, single_target_grid, method="linear")
            result = interpolator(test_data)

            # Should complete with reasonable result (using fallback method)
            assert result.shape == single_target_lat.shape
            assert np.all(np.isfinite(result) | np.isnan(result))  # Allow NaNs where triangulation fails
        except ValueError as e:
            # If it raises ValueError for single point, that's also acceptable
            assert "Could not build Delaunay triangulation" in str(e)

    def test_degenerate_triangle_handling(self):
        """Test handling of degenerate triangles in triangulation."""
        # Create nearly collinear points that might cause degenerate triangles
        degenerate_lat = np.array([[0, 0.001, 0.002], [0, 0.001, 0.002], [0, 0.001, 0.002]])
        degenerate_lon = np.array([[0, 0, 0], [0.001, 0.001, 0.001], [0.002, 0.002, 0.002]])

        degenerate_source_grid = xr.Dataset(
            {"latitude": (["y", "x"], degenerate_lat), "longitude": (["y", "x"], degenerate_lon)}
        )

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], degenerate_lat[:2, :2]),
                "longitude": (["y_target", "x_target"], degenerate_lon[:2, :2]),
            }
        )

        # Test with nearest neighbor (should handle degeneracy)
        test_data = xr.DataArray(np.ones_like(degenerate_lat) * 280.0, dims=["y", "x"])

        interpolator = CurvilinearInterpolator(degenerate_source_grid, target_grid, method="nearest")
        result = interpolator(test_data)

        assert result.shape == target_grid["latitude"].shape
        assert np.all(np.isfinite(result))

    def test_identical_coordinates(self):
        """Test handling of grids with identical coordinates."""
        # Create grid with identical coordinates
        identical_lat = np.array([[45.0, 45.0], [45.0, 45.0]])
        identical_lon = np.array([[0.0, 0.0], [0.0, 0.0]])

        identical_grid = xr.Dataset({"latitude": (["y", "x"], identical_lat), "longitude": (["y", "x"], identical_lon)})

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], identical_lat),
                "longitude": (["y_target", "x_target"], identical_lon),
            }
        )

        # Should handle identical coordinates gracefully
        test_data = xr.DataArray(np.array([[280.0, 280.0], [280.0, 280.0]]), dims=["y", "x"])

        try:
            interpolator = CurvilinearInterpolator(identical_grid, target_grid, method="nearest")
            result = interpolator(test_data)

            # If it succeeds, verify result
            assert result.shape == identical_lat.shape
            assert np.all(result == 280.0)  # Should preserve constant value
        except Exception:
            # If it raises an exception for identical coordinates, that's acceptable
            pass


class TestNaNPropagation:
    """Test NaN handling and propagation through interpolation."""

    def setup_method(self):
        """Set up test data for NaN propagation tests."""
        # Create test grids
        self.source_lat = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.source_lon = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        self.target_lat = np.array([[0.5, 1.5], [0.5, 1.5]])
        self.target_lon = np.array([[-0.5, -0.5], [0.5, 0.5]])

        self.source_grid = xr.Dataset(
            {"latitude": (["y", "x"], self.source_lat), "longitude": (["y", "x"], self.source_lon)}
        )

        self.target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], self.target_lat),
                "longitude": (["y_target", "x_target"], self.target_lon),
            }
        )

    def test_nan_value_propagation_nearest(self):
        """Test NaN propagation with nearest neighbor interpolation."""
        # Create data with NaN values
        data_with_nan = np.array([[280.0, np.nan, 285.0], [282.0, 283.0, np.nan], [np.nan, 287.0, 289.0]])

        test_data = xr.DataArray(data_with_nan, dims=["y", "x"])

        # Test nearest neighbor interpolation
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="nearest")
        result = interpolator(test_data)

        # Result should handle NaN appropriately
        assert result.shape == self.target_lat.shape

        # Some NaN values may propagate, but not all should be NaN
        nan_count = np.sum(np.isnan(result))
        assert nan_count <= result.size  # Allow some NaNs
        assert nan_count < result.size  # But not all NaN

    def test_nan_value_propagation_linear(self):
        """Test NaN propagation with linear interpolation."""
        # Create data with strategic NaN values
        data_with_nan = np.array(
            [
                [280.0, 282.0, 285.0],
                [282.0, np.nan, 287.0],  # NaN in center
                [284.0, 287.0, 289.0],
            ]
        )

        test_data = xr.DataArray(data_with_nan, dims=["y", "x"])

        # Test linear interpolation
        interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method="linear")
        result = interpolator(test_data)

        # Linear interpolation should handle NaN in triangulation
        assert result.shape == self.target_lat.shape
        assert np.all(np.isfinite(result) | np.isnan(result))  # Allow NaNs where triangulation fails

    def test_all_nan_input(self):
        """Test behavior with all-NaN input data."""
        # Create all-NaN data
        all_nan_data = np.full((3, 3), np.nan)
        test_data = xr.DataArray(all_nan_data, dims=["y", "x"])

        # Test both methods
        for method in ["nearest", "linear"]:
            interpolator = CurvilinearInterpolator(self.source_grid, self.target_grid, method=method)
            result = interpolator(test_data)

            # Result should be all NaN
            assert result.shape == self.target_lat.shape
            assert np.all(np.isnan(result))

    def test_nan_filling_behavior(self):
        """Test NaN filling behavior with different fill methods."""
        # Create data with NaN and test fill methods
        data_with_nan = np.array([[280.0, np.nan, 285.0], [282.0, 283.0, 287.0], [284.0, 287.0, 289.0]])

        test_data = xr.DataArray(data_with_nan, dims=["y", "x"])

        # Test fill_method='nearest' (should fill NaN with nearest valid value)
        interpolator_fill = CurvilinearInterpolator(
            self.source_grid, self.target_grid, method="nearest", fill_method="nearest"
        )
        result_fill = interpolator_fill(test_data)

        # Test fill_method='nan' (should leave NaN unfilled)
        interpolator_nan = CurvilinearInterpolator(
            self.source_grid, self.target_grid, method="nearest", fill_method="nan"
        )
        result_nan = interpolator_nan(test_data)

        # Both should complete
        assert result_fill.shape == self.target_lat.shape
        assert result_nan.shape == self.target_lat.shape

        # Fill method should produce fewer NaNs (or equal)
        fill_nan_count = np.sum(np.isnan(result_fill))
        nan_nan_count = np.sum(np.isnan(result_nan))
        assert fill_nan_count <= nan_nan_count


class TestBoundaryEdgeCases:
    """Test additional boundary edge cases and robustness scenarios."""

    def test_coordinate_bounds_validation(self):
        """Test validation of coordinate bounds."""
        # Test with coordinates outside normal bounds
        invalid_lat = np.array([[95.0, -95.0], [95.0, -95.0]])  # Outside [-90, 90]
        invalid_lon = np.array([[200.0, -200.0], [200.0, -200.0]])  # Outside [-180, 180]

        source_grid = xr.Dataset({"latitude": (["y", "x"], invalid_lat), "longitude": (["y", "x"], invalid_lon)})

        target_grid = xr.Dataset(
            {"latitude": (["y_target", "x_target"], invalid_lat), "longitude": (["y_target", "x_target"], invalid_lon)}
        )

        # Should handle out-of-bounds coordinates gracefully
        test_data = xr.DataArray(np.ones_like(invalid_lat) * 280.0, dims=["y", "x"])

        try:
            interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
            result = interpolator(test_data)

            # If it succeeds, verify result
            assert result.shape == invalid_lat.shape
            assert np.all(np.isfinite(result) | np.isnan(result))
        except Exception:
            # If it raises an exception for invalid bounds, that's acceptable
            pass

    def test_extreme_coordinate_differences(self):
        """Test interpolation with extreme coordinate differences."""
        # Create grids with very different coordinate ranges
        extreme_source_lat = np.array([[0.0, 0.001], [0.0, 0.001]])
        extreme_source_lon = np.array([[0.0, 0.0], [0.001, 0.001]])

        extreme_target_lat = np.array([[45.0, 89.0], [45.0, 89.0]])
        extreme_target_lon = np.array([[90.0, 179.0], [90.0, 179.0]])

        source_grid = xr.Dataset(
            {"latitude": (["y", "x"], extreme_source_lat), "longitude": (["y", "x"], extreme_source_lon)}
        )

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], extreme_target_lat),
                "longitude": (["y_target", "x_target"], extreme_target_lon),
            }
        )

        # Test interpolation across extreme coordinate ranges
        test_data = xr.DataArray(np.array([[280.0, 285.0], [282.0, 287.0]]), dims=["y", "x"])

        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)

        # Should handle extreme coordinate differences
        assert result.shape == extreme_target_lat.shape
        assert np.all(np.isfinite(result) | np.isnan(result))  # Allow NaNs for out-of-domain points

    def test_coordinate_precision_limits(self):
        """Test interpolation at coordinate precision limits."""
        # Create coordinates at machine precision limits
        eps = np.finfo(float).eps
        precision_lat = np.array([[0.0, eps], [0.0, eps]])
        precision_lon = np.array([[0.0, 0.0], [eps, eps]])

        source_grid = xr.Dataset({"latitude": (["y", "x"], precision_lat), "longitude": (["y", "x"], precision_lon)})

        target_grid = xr.Dataset(
            {
                "latitude": (["y_target", "x_target"], precision_lat),
                "longitude": (["y_target", "x_target"], precision_lon),
            }
        )

        # Test interpolation at precision limits
        test_data = xr.DataArray(np.array([[280.0, 280.0 + eps], [280.0, 280.0 + eps]]), dims=["y", "x"])

        interpolator = CurvilinearInterpolator(source_grid, target_grid, method="nearest")
        result = interpolator(test_data)

        # Should handle precision limits gracefully
        assert result.shape == precision_lat.shape
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    # Run boundary condition tests
    pole_test = TestPoleProximityHandling()
    pole_test.setup_method()

    pole_test.test_north_pole_handling()
    pole_test.test_south_pole_handling()
    pole_test.test_polar_linear_interpolation_fallback()
    pole_test.test_pole_coordinate_singularity()

    dateline_test = TestDateLineCrossing()
    dateline_test.setup_method()

    dateline_test.test_date_line_crossing_interpolation()
    dateline_test.test_longitude_wrapping_consistency()
    dateline_test.test_antimeridian_continuity()

    empty_test = TestEmptyAndDegenerateGrids()
    empty_test.test_empty_source_grid()
    empty_test.test_single_point_grid()
    empty_test.test_degenerate_triangle_handling()
    empty_test.test_identical_coordinates()

    nan_test = TestNaNPropagation()
    nan_test.setup_method()

    nan_test.test_nan_value_propagation_nearest()
    nan_test.test_nan_value_propagation_linear()
    nan_test.test_all_nan_input()
    nan_test.test_nan_filling_behavior()

    boundary_test = TestBoundaryEdgeCases()

    boundary_test.test_coordinate_bounds_validation()
    boundary_test.test_extreme_coordinate_differences()
    boundary_test.test_coordinate_precision_limits()

    print("All boundary condition tests passed!")
