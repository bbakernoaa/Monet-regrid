import numpy as np
import xarray as xr

from monet_regrid.methods.conservative import neighbor_budget_regrid


def test_neighbor_budget_regrid_simple():
    """Test neighbor-budget regridding with simple, uniform floating-point data."""
    # Create a simple source DataArray with floating-point data
    source_data = np.array(
        [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]]
    )
    source_da = xr.DataArray(
        source_data,
        dims=("y", "x"),
        coords={"y": np.arange(4), "x": np.arange(4)},
    )

    # Create a coarser target grid (2x2)
    target_ds = xr.Dataset(
        coords={"y": np.arange(2) * 2 + 0.5, "x": np.arange(2) * 2 + 0.5}
    )

    # Expected result: each target cell is the average of the 2x2 source cells
    expected_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_da = xr.DataArray(
        expected_data,
        dims=("y", "x"),
        coords={"y": target_ds["y"], "x": target_ds["x"]},
    )

    # Perform the regridding
    regridded_da = neighbor_budget_regrid(source_da, target_ds, n_points=2)

    # Check that the result matches the expected output
    xr.testing.assert_allclose(regridded_da, expected_da)


def test_neighbor_budget_regrid_varied():
    """Test neighbor-budget regridding with varied floating-point data."""
    # Create a source DataArray with varied data to test averaging
    source_data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    )
    source_da = xr.DataArray(
        source_data,
        dims=("y", "x"),
        coords={"y": np.arange(4), "x": np.arange(4)},
    )

    # Create a coarser target grid (2x2)
    target_ds = xr.Dataset(
        coords={"y": np.arange(2) * 2 + 0.5, "x": np.arange(2) * 2 + 0.5}
    )

    # Expected result: average of each 2x2 block
    # Block 1 (top-left): (1+2+5+6)/4 = 3.5
    # Block 2 (top-right): (3+4+7+8)/4 = 5.5
    # Block 3 (bottom-left): (9+10+13+14)/4 = 11.5
    # Block 4 (bottom-right): (11+12+15+16)/4 = 13.5
    expected_data = np.array([[3.5, 5.5], [11.5, 13.5]])
    expected_da = xr.DataArray(
        expected_data,
        dims=("y", "x"),
        coords={"y": target_ds["y"], "x": target_ds["x"]},
    )

    # Perform the regridding
    regridded_da = neighbor_budget_regrid(source_da, target_ds, n_points=2)

    # Check that the result matches the expected output
    xr.testing.assert_allclose(regridded_da, expected_da)
