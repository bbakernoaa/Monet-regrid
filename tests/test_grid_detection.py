import numpy as np
import pytest
import xarray as xr

from monet_regrid.constants import GridType
from monet_regrid.utils import _get_grid_type, validate_grid_compatibility

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old imports: from monet_regrid.constants import ...; from monet_regrid.utils import ...
# New imports: from monet_regrid.constants import ...; from monet_regrid.utils import ...


def test_get_grid_type_rectilinear():
    """Test detection of rectilinear grid type."""
    # Create a simple rectilinear grid
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)

    # Create coordinate arrays
    ds = xr.Dataset(
        coords={
            "latitude": (["latitude"], lat, {"units": "degrees_north"}),
            "longitude": (["longitude"], lon, {"units": "degrees_east"}),
        }
    )

    # Add cf attributes to help cf-xarray
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["longitude"].attrs["standard_name"] = "longitude"

    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.RECTILINEAR


def test_get_grid_type_curvilinear():
    """Test detection of curvilinear grid type."""
    # Create a simple curvilinear grid
    lat_2d = np.random.rand(5, 6)
    lon_2d = np.random.rand(5, 6)

    # Create coordinate arrays
    ds = xr.Dataset(
        coords={
            "latitude": (["y", "x"], lat_2d, {"units": "degrees_north"}),
            "longitude": (["y", "x"], lon_2d, {"units": "degrees_east"}),
        }
    )

    # Add cf attributes to help cf-xarray
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["longitude"].attrs["standard_name"] = "longitude"

    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.CURVILINEAR


def test_get_grid_type_mismatched_dimensions():
    """Test that mismatched coordinate dimensions raise an error."""
    # Create a dataset with mismatched coordinate dimensions
    lat_1d = np.linspace(-90, 90, 10)
    lon_2d = np.random.rand(5, 6)

    ds = xr.Dataset(
        coords={
            "latitude": (["latitude"], lat_1d, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["y", "x"], lon_2d, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    with pytest.raises(ValueError, match="Mismatched coordinate dimensions"):
        _get_grid_type(ds)


def test_get_grid_type_unsupported_dimensions():
    """Test that unsupported coordinate dimensions raise an error."""
    # Create a dataset with 3D coordinates (unsupported)
    lat_3d = np.random.rand(2, 3, 4)
    lon_3d = np.random.rand(2, 3, 4)

    ds = xr.Dataset(
        coords={
            "latitude": (["dim1", "dim2", "dim3"], lat_3d, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["dim1", "dim2", "dim3"], lon_3d, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    with pytest.raises(ValueError, match="Unsupported coordinate dimensions"):
        _get_grid_type(ds)


def test_get_grid_type_missing_coordinates():
    """Test that missing coordinates raise an error."""
    # Create a dataset without proper coordinate information
    ds = xr.Dataset(
        coords={
            "time": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Could not identify coordinate"):
        _get_grid_type(ds)


def test_validate_grid_compatibility():
    """Test grid compatibility validation."""
    # Create two rectilinear grids
    lat1 = np.linspace(-90, 90, 10)
    lon1 = np.linspace(-180, 180, 20)

    ds1 = xr.Dataset(
        coords={
            "latitude": (["latitude"], lat1, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["longitude"], lon1, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    lat2 = np.linspace(-45, 45, 5)
    lon2 = np.linspace(-90, 90, 10)

    ds2 = xr.Dataset(
        coords={
            "latitude": (["latitude"], lat2, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["longitude"], lon2, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    source_type, target_type = validate_grid_compatibility(ds1, ds2)
    assert source_type == GridType.RECTILINEAR
    assert target_type == GridType.RECTILINEAR


def test_validate_grid_compatibility_curvilinear():
    """Test grid compatibility validation for curvilinear grids."""
    # Create two curvilinear grids
    lat1_2d = np.random.rand(5, 6)
    lon1_2d = np.random.rand(5, 6)

    ds1 = xr.Dataset(
        coords={
            "latitude": (["y", "x"], lat1_2d, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["y", "x"], lon1_2d, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    lat2_2d = np.random.rand(3, 4)
    lon2_2d = np.random.rand(3, 4)

    ds2 = xr.Dataset(
        coords={
            "latitude": (["y", "x"], lat2_2d, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["y", "x"], lon2_2d, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    source_type, target_type = validate_grid_compatibility(ds1, ds2)
    assert source_type == GridType.CURVILINEAR
    assert target_type == GridType.CURVILINEAR


def test_validate_grid_compatibility_mixed():
    """Test grid compatibility validation for mixed grid types."""
    # Create a rectilinear grid
    lat1 = np.linspace(-90, 90, 10)
    lon1 = np.linspace(-180, 180, 20)

    ds1 = xr.Dataset(
        coords={
            "latitude": (["latitude"], lat1, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["longitude"], lon1, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    # Create a curvilinear grid
    lat2_2d = np.random.rand(5, 6)
    lon2_2d = np.random.rand(5, 6)

    ds2 = xr.Dataset(
        coords={
            "latitude": (["y", "x"], lat2_2d, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["y", "x"], lon2_2d, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    source_type, target_type = validate_grid_compatibility(ds1, ds2)
    assert source_type == GridType.RECTILINEAR
    assert target_type == GridType.CURVILINEAR


def test_get_grid_type_rectilinear_alternative_names():
    """Test detection of rectilinear grid with alternative coordinate names."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    ds = xr.Dataset(coords={"lat": (["lat"], lat), "lon": (["lon"], lon)})
    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.RECTILINEAR


def test_get_grid_type_curvilinear_alternative_names():
    """Test detection of curvilinear grid with alternative coordinate names."""
    lat_2d = np.random.rand(5, 6)
    lon_2d = np.random.rand(5, 6)
    ds = xr.Dataset(coords={"YC": (["y", "x"], lat_2d), "XC": (["y", "x"], lon_2d)})
    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.CURVILINEAR


def test_get_grid_type_rectilinear_no_cf_attrs():
    """Test rectilinear detection without cf attributes."""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.RECTILINEAR


def test_get_grid_type_curvilinear_no_cf_attrs():
    """Test curvilinear detection without cf attributes."""
    lat_2d = np.random.rand(5, 6)
    lon_2d = np.random.rand(5, 6)
    ds = xr.Dataset(coords={"latitude": (("y", "x"), lat_2d), "longitude": (("y", "x"), lon_2d)})
    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.CURVILINEAR


def test_get_grid_type_curvilinear_by_units():
    """Test curvilinear detection by units when names are non-standard."""
    lat_2d = np.random.rand(5, 6)
    lon_2d = np.random.rand(5, 6)
    ds = xr.Dataset(
        coords={
            "some_lat_var": (
                ("y", "x"),
                lat_2d,
                {"units": "degrees_north"},
            ),
            "some_lon_var": (
                ("y", "x"),
                lon_2d,
                {"units": "degrees_east"},
            ),
        }
    )
    grid_type = _get_grid_type(ds)
    assert grid_type == GridType.CURVILINEAR


def test_get_grid_type_missing_one_coord():
    """Test that an error is raised if only one coordinate is found."""
    lat = np.linspace(-90, 90, 10)
    ds = xr.Dataset(coords={"latitude": lat})
    with pytest.raises(ValueError, match="Could not identify coordinate: Could not identify longitude coordinate"):
        _get_grid_type(ds)
