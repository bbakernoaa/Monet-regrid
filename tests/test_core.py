
import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import BaseRegridder

class MockRegridder(BaseRegridder):
    def __call__(self):
        pass

    def to_file(self, filepath: str):
        pass

    @classmethod
    def from_file(cls, filepath: str):
        pass

    def info(self):
        pass

    def _validate_inputs(self) -> None:
        pass

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return xr.Dataset(
        {
            "data": (("lat", "lon"), np.random.rand(10, 20)),
        },
        coords={
            "lat": np.arange(10),
            "lon": np.arange(20),
        },
    )

def test_identify_coords_by_name(sample_dataset):
    """Test that coordinates are identified by name."""
    regridder = MockRegridder(sample_dataset, sample_dataset)
    lat_coords = regridder._identify_lat_coords(sample_dataset)
    lon_coords = regridder._identify_lon_coords(sample_dataset)
    assert lat_coords == ["lat"]
    assert lon_coords == ["lon"]

def test_identify_coords_by_alternative_name():
    """Test that coordinates are identified by alternative names like 'y' and 'x'."""
    ds = xr.Dataset(
        {
            "data": (("y", "x"), np.random.rand(10, 20)),
        },
        coords={
            "y": np.arange(10),
            "x": np.arange(20),
        },
    )
    regridder = MockRegridder(ds, ds)
    lat_coords = regridder._identify_lat_coords(ds)
    lon_coords = regridder._identify_lon_coords(ds)
    assert lat_coords == ["y"]
    assert lon_coords == ["x"]

def test_identify_coords_by_dim():
    """Test that coordinates are identified by dimension when coord name doesn't match."""
    # Create a DataArray with dimensions 'lat' and 'lon'
    data = np.random.rand(10, 20)
    dims = ("lat", "lon")

    # Create coordinates with names that do NOT match the keywords for lat/lon
    coords = {
        "coord1": ("lat", np.arange(10)),
        "coord2": ("lon", np.arange(20)),
    }

    ds = xr.Dataset({"data": (dims, data)}, coords=coords)
    regridder = MockRegridder(ds, ds)

    lat_coords = regridder._identify_lat_coords(ds)
    lon_coords = regridder._identify_lon_coords(ds)

    assert lat_coords == ["lat"]
    assert lon_coords == ["lon"]

def test_identify_coords_by_cf_xarray(sample_dataset):
    """Test that coordinates are identified by cf-xarray when name doesn't match."""
    ds = sample_dataset.rename({"lat": "y_coord", "lon": "x_coord"})
    ds.y_coord.attrs["standard_name"] = "latitude"
    ds.x_coord.attrs["standard_name"] = "longitude"
    regridder = MockRegridder(ds, ds)
    lat_coords = regridder._identify_lat_coords(ds)
    lon_coords = regridder._identify_lon_coords(ds)
    assert lat_coords == ["y_coord"]
    assert lon_coords == ["x_coord"]

def test_identify_coords_no_match():
    """Test that no coordinates are identified when there is no match."""
    ds = xr.Dataset({"data": (("z", "t"), np.random.rand(10, 20))})
    regridder = MockRegridder(ds, ds)
    lat_coords = regridder._identify_lat_coords(ds)
    lon_coords = regridder._identify_lon_coords(ds)
    assert lat_coords == []
    assert lon_coords == []
