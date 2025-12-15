import numpy as np
import xarray as xr
import monet_regrid  # noqa: F401


def test_curvilinear_linear_vs_nearest_differentiation():
    """Test that linear interpolation on a sphere produces different results from nearest neighbor.

    This ensures that the linear interpolation doesn't silently fallback to nearest neighbor
    due to points being slightly outside the convex hull on spherical geometries.
    """
    # Create sample curvilinear source grid
    ny_curv, nx_curv = 50, 100
    lon_1d = np.linspace(-180, 180, nx_curv)
    lat_1d = np.linspace(-90, 90, ny_curv)
    source_lon_2d, source_lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Add some curvilinear distortion to make it non-rectilinear
    source_lon_2d = source_lon_2d + 0.5 * np.sin(np.radians(source_lat_2d)) * np.cos(np.radians(source_lon_2d))
    source_lat_2d = source_lat_2d + 0.3 * np.cos(np.radians(source_lat_2d)) * np.sin(np.radians(source_lon_2d))

    source_ds_curv = xr.Dataset(
        {"temperature": (("y", "x"), np.random.random((ny_curv, nx_curv)).astype(np.float32), {"units": "K"})},
        coords={"lon": (("y", "x"), source_lon_2d), "lat": (("y", "x"), source_lat_2d)},
    )

    # Create sample curvilinear target grid
    ny_target, nx_target = 40, 80
    lon_1d_target = np.linspace(-180, 180, nx_target)
    lat_1d_target = np.linspace(-90, 90, ny_target)
    target_lon_2d, target_lat_2d = np.meshgrid(lon_1d_target, lat_1d_target)

    # Add different distortion to target
    target_lon_2d = target_lon_2d + 0.3 * np.sin(np.radians(target_lat_2d)) * np.cos(np.radians(target_lon_2d))
    target_lat_2d = target_lat_2d + 0.2 * np.cos(np.radians(target_lat_2d)) * np.sin(np.radians(target_lon_2d))

    target_ds_curv = xr.Dataset(coords={"lon": (("y", "x"), target_lon_2d), "lat": (("y", "x"), target_lat_2d)})

    # 1. Run Linear Interpolation
    result_linear = source_ds_curv["temperature"].regrid.linear(target_ds_curv)

    # 2. Run Nearest Neighbor Interpolation
    result_nearest = source_ds_curv["temperature"].regrid.nearest(target_ds_curv)

    # 3. Compare Results
    # They should be different
    diff = np.abs(result_linear - result_nearest)
    max_diff = diff.max().values

    # Ideally, max difference should be significant (e.g. > 0.1 for random [0,1] data)
    assert max_diff > 0.01, "Linear and Nearest Neighbor results are too similar (likely identical)"


def test_curvilinear_scaling_recovery():
    """Test that the scaling mechanism actually recovers points that would otherwise be lost."""
    # Create a simple spherical case where we know points are "outside" the convex hull
    # 4 points on equator forming a square (in lat/lon) -> tetrahedron in 3D? No, just surface points.

    # Use a coarse grid where the chord vs arc difference is large
    ny, nx = 10, 20
    lon = np.linspace(0, 360, nx, endpoint=False)
    lat = np.linspace(-90, 90, ny)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    source_ds = xr.Dataset(coords={"lon": (("y", "x"), lon_2d), "lat": (("y", "x"), lat_2d)})

    # Target points exactly in between source points on the sphere surface
    # These will definitely be "outside" the planar facets of the source convex hull
    ny_t, nx_t = 5, 10
    lon_t = np.linspace(10, 350, nx_t)
    lat_t = np.linspace(-80, 80, ny_t)
    lon_t_2d, lat_t_2d = np.meshgrid(lon_t, lat_t)

    target_ds = xr.Dataset(coords={"lon": (("y", "x"), lon_t_2d), "lat": (("y", "x"), lat_t_2d)})

    test_data = xr.DataArray(
        np.random.rand(ny, nx),
        dims=["y", "x"],
        coords={"lon": (("y", "x"), lon_2d), "lat": (("y", "x"), lat_2d)},
    )
    result = test_data.regrid.linear(target_ds)

    # Without scaling, this would be 0. With scaling, it should be high.
    assert not np.all(np.isnan(result))
