import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def test_curvilinear_longitude_alignment_stat():
    """Test that curvilinear statistical regridding correctly aligns longitude ranges.

    Source curvilinear grid uses [180, 360] range.
    Target rectilinear grid uses [-180, 180] range.
    Without proper alignment, this would result in NaNs for the overlapping region.
    """
    # 1. Create source curvilinear grid [180, 360]
    # We'll make it a simple "rectilinear-as-curvilinear" for easy verification
    lon_1d = np.linspace(190, 350, 10)
    lat_1d = np.linspace(-40, 40, 10)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    source_da = xr.DataArray(
        np.ones((10, 10)),
        dims=["y", "x"],
        coords={
            "longitude": (("y", "x"), lon_2d),
            "latitude": (("y", "x"), lat_2d),
        },
        name="test_data",
    )

    # 2. Create target rectilinear grid [-180, 180]
    # This overlaps with 190-350 if we shift it (190-350 -> -170 to -10)
    target_ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.linspace(-30, 30, 5)),
            "lon": (["lon"], np.linspace(-170, -10, 5)),
        }
    )

    # 3. Regrid using stat
    regridder = CurvilinearRegridder(source_da, target_ds)
    result = regridder.stat(method="mean")

    # 4. Verify results
    # If alignment worked, the mean should be 1.0 (since all source is 1.0)
    # If alignment failed, we would get NaNs because [190, 350] is outside [-170, -10]
    assert not result.isnull().all(), "Result should not be all NaNs; longitude alignment failed."
    np.testing.assert_allclose(result.values, 1.0)

    # Verify history
    assert "Pre-formatted" in result.attrs["history"]
    assert "statistic_reduce" in result.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
