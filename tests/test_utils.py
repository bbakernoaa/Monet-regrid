
import numpy as np
import xarray as xr

from monet_regrid.utils import format_lat


def test_format_lat():
    """Test the format_lat function."""
    ds = xr.Dataset(
        coords={
            "latitude": np.arange(-89, 90, 1),
            "longitude": np.arange(-180, 181, 1),
        }
    )
    ds["data"] = (("latitude", "longitude"), np.random.rand(179, 361))
    ds.attrs["history"] = "testing"
    formatted_coords = {"lat": "latitude", "lon": "longitude"}
    # Create a dummy target dataset, it's not used by format_lat
    target = xr.Dataset()
    result = format_lat(ds, target, formatted_coords)
    assert result.latitude[0] == -90
    assert result.latitude[-1] == 90
    assert result.data.shape == (181, 361)
    assert "history" in result.attrs
