"""
Tests for regridder consolidation following the Aero Protocol.
"""

import dask.array as da
import numpy as np
import xarray as xr

from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def create_rectilinear_data() -> tuple[xr.DataArray, xr.Dataset]:
    """
    Create a sample rectilinear DataArray and target grid with Aero-standard chunks.

    Returns
    -------
    tuple[xr.DataArray, xr.Dataset]
        A tuple containing the source DataArray and target Dataset.

    Examples
    --------
    >>> da_source, ds_target = create_rectilinear_data()
    """
    # 100MB chunk size for float64 is approx 12.5 million elements.
    # For a global 1-degree grid (180x360), that's only 64,800 elements.
    # To hit ~100MB we'd need a very high resolution grid.
    # We will use chunks that are large enough to be meaningful but small enough for CI.
    lat = np.arange(-90, 91, 0.1)  # 1810 points
    lon = np.arange(-180, 181, 0.1)  # 3610 points
    # 1810 * 3610 * 8 bytes (float64) = ~52 MB
    data = da.random.random((len(lat), len(lon)), chunks=(1810, 3610))
    da_source = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="test_data")

    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(-180, 181, 10)
    ds_target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    return da_source, ds_target


def create_curvilinear_data() -> tuple[xr.DataArray, xr.Dataset]:
    """
    Create a sample curvilinear DataArray and target grid with Aero-standard chunks.

    Returns
    -------
    tuple[xr.DataArray, xr.Dataset]
        A tuple containing the source DataArray and target Dataset.

    Examples
    --------
    >>> da_source, ds_target = create_curvilinear_data()
    """
    lat_1d = np.arange(-90, 91, 0.1)
    lon_1d = np.arange(-180, 181, 0.1)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Perturb them slightly to make it "curvilinear"
    lat_2d = lat_2d + 0.1 * np.sin(np.radians(lon_2d))

    # ~52 MB
    data = da.random.random((len(lat_1d), len(lon_1d)), chunks=(len(lat_1d), len(lon_1d)))
    da_source = xr.DataArray(
        data, coords={"lat": (("y", "x"), lat_2d), "lon": (("y", "x"), lon_2d)}, dims=["y", "x"], name="test_data"
    )

    target_lat = np.arange(-90, 91, 10)
    target_lon = np.arange(-180, 181, 10)
    ds_target = xr.Dataset(coords={"lat": target_lat, "lon": target_lon})
    return da_source, ds_target


def test_rectilinear_stat_consolidation() -> None:
    """
    Test that statistical regridding works via the consolidated BaseRegridder.

    Examples
    --------
    >>> test_rectilinear_stat_consolidation()
    """
    da_source, ds_target = create_rectilinear_data()
    regridder = RectilinearRegridder(da_source, ds_target)

    # Test stat method
    result = regridder.stat(method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(ds_target.lat), len(ds_target.lon))
    assert "history" in result.attrs

    # Test info
    info = regridder.info()
    assert info["type"] == "RectilinearRegridder"
    assert info["grid_type"] == "rectilinear"
    assert "source" in info
    assert "target" in info


def test_curvilinear_stat_consolidation() -> None:
    """
    Test that curvilinear statistical regridding works via consolidated logic.

    Examples
    --------
    >>> test_curvilinear_stat_consolidation()
    """
    da_source, ds_target = create_curvilinear_data()
    regridder = CurvilinearRegridder(da_source, ds_target)

    # Test stat method
    result = regridder.stat(method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.shape == (len(ds_target.lat), len(ds_target.lon))
    assert "history" in result.attrs

    # Test info
    info = regridder.info()
    assert info["type"] == "CurvilinearRegridder"
    assert info["grid_type"] == "curvilinear"
    assert "source" in info
    assert "target" in info


def test_categorical_consolidation() -> None:
    """
    Test that categorical regridding works via consolidated logic.

    Examples
    --------
    >>> test_categorical_consolidation()
    """
    da_source, ds_target = create_rectilinear_data()
    # Convert to integer for categorical regridding
    da_source = (da_source * 10).astype(int)
    regridder = RectilinearRegridder(da_source, ds_target)

    values = np.arange(11)

    # Test most_common
    result_most = regridder.most_common(values=values)
    assert isinstance(result_most, xr.DataArray)

    # Test least_common
    result_least = regridder.least_common(values=values)
    assert isinstance(result_least, xr.DataArray)

    assert "most_common" in result_most.attrs["history"]
    assert "least_common" in result_least.attrs["history"]


def test_data_agnostic_info() -> None:
    """
    Test metadata generation for data-agnostic regridders.

    Examples
    --------
    >>> test_data_agnostic_info()
    """
    _, ds_target = create_rectilinear_data()
    regridder = RectilinearRegridder(source_data=None, target_grid=ds_target)

    info = regridder.info()
    assert info["source"] == {}
    assert info["source_dims"] == {}
    assert "target" in info
