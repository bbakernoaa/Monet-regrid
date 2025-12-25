
import numpy as np
import pytest
import xarray as xr

from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def test_rectilinear_regridder_history_preservation():
    """Test that the history attribute is updated correctly when one already exists."""
    source_da = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={"lat": (("y",), np.linspace(0, 1, 10)), "lon": (("x",), np.linspace(0, 1, 20))},
        attrs={"history": "Initial state."},
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y",), np.linspace(0, 1, 5)),
            "lon": (("x",), np.linspace(0, 1, 10)),
        },
    )
    regridder = RectilinearRegridder(source_data=source_da, target_grid=target_ds)
    regridded_da = regridder(method="linear")

    expected_history = "Initial state.\nRegridded using RectilinearRegridder with method='linear'"
    assert "history" in regridded_da.attrs
    assert regridded_da.attrs["history"] == expected_history


def test_rectilinear_regridder_history_creation():
    """Test that the history attribute is created if it does not exist."""
    source_da = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={"lat": (("y",), np.linspace(0, 1, 10)), "lon": (("x",), np.linspace(0, 1, 20))},
    )
    target_ds = xr.Dataset(
        coords={
            "lat": (("y",), np.linspace(0, 1, 5)),
            "lon": (("x",), np.linspace(0, 1, 10)),
        },
    )
    regridder = RectilinearRegridder(source_data=source_da, target_grid=target_ds)
    regridded_da = regridder(method="nearest")

    expected_history = "Regridded using RectilinearRegridder with method='nearest'"
    assert "history" in regridded_da.attrs
    assert regridded_da.attrs["history"] == expected_history


def test_curvilinear_regridder_history_preservation():
    """Test history attribute update for the curvilinear regridder when one exists."""
    lon = np.arange(5, 15, 2)
    lat = np.arange(40, 50, 2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    source_da = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        coords={"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
        attrs={"history": "Curvilinear initial state."},
    )
    target_ds = xr.Dataset(coords={"lat": np.arange(40, 50, 1), "lon": np.arange(5, 15, 1)})
    regridder = CurvilinearRegridder(source_data=source_da, target_grid=target_ds)
    regridded_da = regridder(method="linear")

    expected_history = "Curvilinear initial state.\nRegridded using CurvilinearRegridder with method='linear'"
    assert "history" in regridded_da.attrs
    assert regridded_da.attrs["history"] == expected_history


def test_curvilinear_regridder_history_creation():
    """Test that the history attribute is created for the curvilinear regridder if it does not exist."""
    lon = np.arange(5, 15, 2)
    lat = np.arange(40, 50, 2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    source_da = xr.DataArray(
        np.random.rand(5, 5),
        dims=["y", "x"],
        coords={"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
    )
    target_ds = xr.Dataset(coords={"lat": np.arange(40, 50, 1), "lon": np.arange(5, 15, 1)})
    regridder = CurvilinearRegridder(source_data=source_da, target_grid=target_ds)
    regridded_da = regridder(method="nearest")

    expected_history = "Regridded using CurvilinearRegridder with method='nearest'"
    assert "history" in regridded_da.attrs
    assert regridded_da.attrs["history"] == expected_history
