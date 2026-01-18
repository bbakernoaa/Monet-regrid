import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder


def test_create_source_grid_from_data_lazy_fallback():
    """Test that lazy coordinates are generated when no explicit coords are present."""
    # Create a Dask-backed DataArray without explicit lat/lon coordinates
    dask_data = da.random.random((10, 20), chunks=(5, 10))
    source_da = xr.DataArray(dask_data, dims=["y", "x"])

    # A target grid is required to initialize the regridder
    target_ds = xr.Dataset(
        coords={
            "lat": (("y_new",), np.arange(0.5, 10, 2)),
            "lon": (("x_new",), np.arange(0.5, 20, 2)),
        }
    )

    # Initialize the regridder (source_data is not used in the method being tested)
    regridder = CurvilinearRegridder(source_data=None, target_grid=target_ds)

    # Call the method to generate the source grid
    source_grid = regridder._create_source_grid_from_data(source_da)

    # The Proof: Assert that the generated coordinates are Dask arrays
    assert "latitude" in source_grid.coords
    assert "longitude" in source_grid.coords
    assert isinstance(source_grid["latitude"].data, da.Array)
    assert isinstance(source_grid["longitude"].data, da.Array)
    assert source_grid["latitude"].shape == (10, 20)
    assert source_grid["longitude"].shape == (10, 20)
    assert source_grid["latitude"].dims == ("y", "x")
    assert source_grid["longitude"].dims == ("y", "x")
