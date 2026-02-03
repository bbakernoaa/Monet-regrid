import dask.array as da
import numpy as np

from monet_regrid.coordinate_transformer import WGS84_TO_3D, CoordinateTransformer
from monet_regrid.interpolation.core import InterpolationEngine


def test_coordinate_transformer_types():
    """Verify CoordinateTransformer handles both numpy and dask arrays with correct types."""
    transformer = CoordinateTransformer()

    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])

    # Numpy path
    x, y, z = transformer.transform_coordinates(lon, lat)
    assert isinstance(x, np.ndarray)
    assert x.shape == lon.shape

    # Dask path
    lon_da = da.from_array(lon, chunks=1)
    lat_da = da.from_array(lat, chunks=1)
    xd, yd, zd = transformer.transform_coordinates(lon_da, lat_da)
    assert isinstance(xd, da.Array)
    assert xd.compute().shape == lon.shape


def test_interpolation_engine_build():
    """Verify InterpolationEngine can build structures for various methods."""
    engine = InterpolationEngine(method="nearest")

    source_points = np.random.rand(10, 3)
    target_points = np.random.rand(5, 3)

    engine.build_structures(source_points, target_points)
    assert engine.source_kdtree is not None

    # Apply interpolation
    source_data = np.random.rand(10)
    result = engine.interpolate(source_data)
    assert result.shape == (5,)


def test_interpolation_engine_linear_grid():
    """Verify InterpolationEngine handles linear grid-based interpolation."""
    engine = InterpolationEngine(method="linear")

    # Create a small grid in geocentric coordinates
    lon = np.linspace(0, 10, 4)
    lat = np.linspace(0, 10, 4)
    lon2d, lat2d = np.meshgrid(lon, lat)

    x, y, z = WGS84_TO_3D.transform_coordinates(lon2d, lat2d)
    source_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Target point in the middle
    tx, ty, tz = WGS84_TO_3D.transform_coordinates(np.array([5.0]), np.array([5.0]))
    target_points = np.stack([tx, ty, tz], axis=1)

    engine.build_structures(source_points, target_points, source_shape=(4, 4))

    source_data = lon2d.flatten()
    result = engine.interpolate(source_data)
    # The middle value should be around 5.0 (interpolated from lon)
    assert np.allclose(result, 5.0, atol=0.1)
