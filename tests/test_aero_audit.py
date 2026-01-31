import numpy as np
import pandas as pd
import xarray as xr

from monet_regrid.utils import format_for_regrid, overlap


def test_overlap_optimized():
    """Verify that the optimized overlap function returns correct results."""
    # Source intervals
    a = pd.IntervalIndex.from_breaks([0, 1, 2, 3])
    # Target intervals
    b = pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5])

    # Expected overlap (N=3, M=2)
    # a[0] (0-1) overlaps b[0] (0.5-1.5) by 0.5
    # a[0] (0-1) overlaps b[1] (1.5-2.5) by 0
    # a[1] (1-2) overlaps b[0] (0.5-1.5) by 0.5
    # a[1] (1-2) overlaps b[1] (1.5-2.5) by 0.5
    # a[2] (2-3) overlaps b[0] (0.5-1.5) by 0
    # a[2] (2-3) overlaps b[1] (1.5-2.5) by 0.5

    res = overlap(a, b)
    assert res.shape == (3, 2)
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    np.testing.assert_allclose(res, expected)


def test_overlap_optimized_swapped():
    """Verify overlap with len(a) < len(b) (triggers optimized path)."""
    a = pd.IntervalIndex.from_breaks([0, 1])
    b = pd.IntervalIndex.from_breaks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    res = overlap(a, b)
    assert res.shape == (1, 5)
    expected = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    np.testing.assert_allclose(res, expected)


def test_history_provenance():
    """Verify that format_for_regrid updates the history attribute."""
    ds = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-80, 81, 20), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0, 360, 20), {"units": "degrees_east"}),
        }
    )
    # Mock data
    ds["data"] = (["lat", "lon"], np.random.rand(len(ds.lat), len(ds.lon)))

    target = xr.Dataset(
        coords={
            "lat": (["lat"], np.arange(-90, 91, 10), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(-180, 181, 10), {"units": "degrees_east"}),
        }
    )

    res = format_for_regrid(ds, target)
    assert "history" in res.attrs
    assert "Pre-formatted" in res.attrs["history"]
