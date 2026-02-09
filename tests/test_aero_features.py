from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import RectilinearRegridder


def test_rectilinear_lazy_coordinate_generation():
    """Test that RectilinearRegridder can handle data without explicit coordinates."""
    # Create data WITHOUT coordinates, only dimensions
    # Using dask to ensure it stays lazy
    # We use matching names for target to avoid expansion
    data_values = da.random.random((10, 20), chunks=(5, 5))
    data = xr.DataArray(data_values, dims=["lat", "lon"])

    target_grid = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(0, 9, 5)),
            "lon": (("lon",), np.linspace(0, 19, 10)),
        }
    )

    # RectilinearRegridder should now accept this
    regridder = RectilinearRegridder(data, target_grid, method="linear")

    # Execute regridding
    regridded = regridder()

    # Verify result
    assert regridded.shape == (5, 10)
    assert "lat" in regridded.coords
    assert "lon" in regridded.coords
    assert "history" in regridded.attrs
    assert "Regridded using RectilinearRegridder" in regridded.attrs["history"]


def test_visualize_static_mock():
    """Test the visualize method Track A (static) with mocks."""
    data = xr.DataArray(
        np.random.rand(10, 20), dims=["lat", "lon"], coords={"lat": np.arange(10), "lon": np.arange(20)}, name="test_var"
    )

    mock_ax = MagicMock()
    # Use patch.dict on sys.modules to mock missing top-level modules
    mock_plt = MagicMock()
    mock_ccrs = MagicMock()
    modules = {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt, "cartopy": MagicMock(), "cartopy.crs": mock_ccrs}

    with patch.dict("sys.modules", modules):
        # We also need to mock the plot method of DataArray
        with patch.object(xr.DataArray, "plot") as mock_plot:
            # We need to ensure identify_cf_coordinates is not failing
            with patch("monet_regrid.accessor.identify_cf_coordinates", return_value=("lat", "lon")):
                # Pass ax explicitly to avoid the subplots unpacking issue in mock environment
                data.regrid.visualize(mode="static", ax=mock_ax)

            # Check if plot was called with correct arguments
            mock_plot.assert_called_once()
            _, plot_kwargs = mock_plot.call_args
            assert plot_kwargs["x"] == "lon"
            assert plot_kwargs["y"] == "lat"
            assert "transform" in plot_kwargs


def test_visualize_interactive_mock():
    """Test the visualize method Track B (interactive) with mocks."""
    data = xr.DataArray(
        np.random.rand(10, 20), dims=["lat", "lon"], coords={"lat": np.arange(10), "lon": np.arange(20)}, name="test_var"
    )

    # We need to mock the hvplot accessor
    modules = {"hvplot": MagicMock(), "hvplot.xarray": MagicMock()}
    with patch.dict("sys.modules", modules):
        mock_hvplot = MagicMock()
        with patch.object(xr.DataArray, "hvplot", mock_hvplot, create=True):
            data.regrid.visualize(mode="interactive", rasterize=True)

            # Check if hvplot was called with Aero defaults
            mock_hvplot.assert_called_once()
            _, kwargs = mock_hvplot.call_args
            assert kwargs["x"] == "lon"
            assert kwargs["y"] == "lat"
            assert kwargs["rasterize"] is True
            assert kwargs["geo"] is True


def test_identify_cf_coordinates_standard():
    """Test that identify_cf_coordinates finds coordinates with standard names."""
    from monet_regrid.utils import identify_cf_coordinates

    # Standard names in coords
    ds = xr.Dataset(
        coords={"lat": (("lat",), np.arange(10)), "lon": (("lon",), np.arange(10))},
        data_vars={"a": (("lat", "lon"), np.zeros((10, 10)))},
    )
    lat_name, lon_name = identify_cf_coordinates(ds)
    assert lat_name == "lat"
    assert lon_name == "lon"


def test_visualize_unsupported_mode():
    """Test that visualize raises error for unsupported mode."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"])
    with pytest.raises(ValueError, match="Unknown mode"):
        data.regrid.visualize(mode="invalid")
