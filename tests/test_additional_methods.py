"""Tests for bilinear and cubic interpolation methods."""

import numpy as np
import xarray as xr

import monet_regrid  # noqa: F401


def create_projected_grid(nx, ny, x_range, y_range):
    """Create a grid that is rectilinear in Geocentric X,Y space (near North Pole)."""
    r = 6371000.0  # Approx Earth radius in meters

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x, y)

    # Calculate Z to be on sphere
    r2 = xx**2 + yy**2
    z = np.sqrt(r**2 - r2)

    # Convert to Lat/Lon
    # lat = asin(z/r)
    # lon = atan2(y, x)
    lat_rad = np.arcsin(z / r)
    lon_rad = np.arctan2(yy, xx)

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    ds = xr.Dataset({"latitude": (["y", "x"], lat_deg), "longitude": (["y", "x"], lon_deg)})
    return ds, xx, yy


def test_bilinear_interpolation():
    """Test bilinear interpolation."""
    # Create source grid rectilinear in projected space
    # 20km x 20km box at North Pole
    source_grid, sx, sy = create_projected_grid(5, 5, (-10000, 10000), (-10000, 10000))

    # Create target grid (finer)
    target_grid, tx, ty = create_projected_grid(9, 9, (-10000, 10000), (-10000, 10000))

    # Create test data (linear in X and Y)
    # Data = a*X + b*Y
    # Since grid is rectilinear in X,Y, bilinear interpolation should be exact
    data_values = (sx + sy) / 1000.0  # Scale down
    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={
            "latitude": (("y", "x"), source_grid.latitude.data),
            "longitude": (("y", "x"), source_grid.longitude.data),
        },
    )

    # Interpolate
    result = test_data.regrid.bilinear(target_grid)

    # Check result
    expected = (tx + ty) / 1000.0

    # Should be very close (floating point errors only)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5, atol=1e-5)


def test_cubic_interpolation():
    """Test cubic interpolation."""
    # Create source grid rectilinear in projected space
    source_grid, sx, sy = create_projected_grid(10, 10, (-20000, 20000), (-20000, 20000))

    # Create target grid
    target_grid, tx, ty = create_projected_grid(10, 10, (-10000, 10000), (-10000, 10000))

    # Create test data (cubic in X and Y)
    # Data = (X/1000)^3 + (Y/1000)^3
    data_values = (sx / 10000.0) ** 3 + (sy / 10000.0) ** 3
    test_data = xr.DataArray(
        data_values,
        dims=["y", "x"],
        coords={
            "latitude": (("y", "x"), source_grid.latitude.data),
            "longitude": (("y", "x"), source_grid.longitude.data),
        },
    )

    # Interpolate
    result = test_data.regrid.cubic(target_grid)

    # Check result
    expected = (tx / 10000.0) ** 3 + (ty / 10000.0) ** 3

    # Cubic interpolation should be reasonably accurate
    # Focus on center to avoid boundary artifacts
    center_slice = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(result.values[center_slice], expected[center_slice], rtol=0.1, atol=0.1)


if __name__ == "__main__":
    test_bilinear_interpolation()
    test_cubic_interpolation()
