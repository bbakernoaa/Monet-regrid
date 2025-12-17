#!/usr/bin/env python
"""Test script to verify the RASM dataset coordinate validation fix."""
import logging
import numpy as np
import xarray as xr

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: import monet_regrid  # Import to register the accessor
# New import: import monet_regrid  # Import to register the accessor


def test_rasm_coordinate_validation():
    """Test the exact scenario from the user's issue with RASM dataset."""
    logging.info("Testing RASM dataset coordinate validation fix...")

    # Load RASM dataset (curvilinear grid with xc, yc coordinates)
    logging.info("Loading RASM dataset...")
    ds = xr.tutorial.open_dataset("rasm")
    logging.info("✓ RASM dataset loaded successfully")
    logging.info("  Dimensions: %s", ds.dims)
    logging.info("  Coordinates: %s", list(ds.coords))
    logging.info("  Data variables: %s", list(ds.data_vars))

    # Create rectilinear target grid with lat/lon coordinates
    logging.info("\nCreating rectilinear target grid...")
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(16, 75, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(200, 330, 1.5), {"units": "degrees_east"}),
        }
    )
    logging.info("✓ Target grid created")
    logging.info("  Dimensions: %s", ds_out.dims)
    logging.info("  Coordinates: %s", list(ds_out.coords))

    # This should now work without ValueError
    logging.info("\nTesting build_regridder with RASM dataset...")
    regridder = ds.regrid.build_regridder(ds_out, method="linear")
    logging.info("✓ Success: Regridder created successfully")
    logging.info("  Regridder type: %s", type(regridder).__name__)
    logging.info("  Regridder info: %s", regridder.info())

    # Test that the regridder can be applied to data
    logging.info("\nTesting regridder application...")
    # Use one of the data variables from RASM dataset
    var_names = list(ds.data_vars)  # Get data variable names
    var_name = var_names[0]  # Get first data variable name
    logging.info("  Using variable: %s", var_name)

    # Apply regridding to a single variable
    result = ds[var_name].regrid.linear(ds_out)
    logging.info("✓ Success: Data regridded successfully")
    logging.info(" Original shape: %s", ds[var_name].shape)
    logging.info(" Regridded shape: %s", result.shape)
    logging.info(" Result coordinates: %s", list(result.coords))

    # Test with the full dataset
    logging.info("\nTesting regridder with full dataset...")
    result_ds = ds.regrid.linear(ds_out)
    logging.info("✓ Success: Full dataset regridded successfully")
    logging.info("  Result variables: %s", list(result_ds.data_vars))

    logging.info("\n%s", "=" * 60)
    logging.info("ALL TESTS PASSED! RASM coordinate validation fix works correctly.")
    logging.info("=" * 60)


def test_coordinate_mapping():
    """Test that coordinate mapping works properly (xc->lon, yc->lat)."""
    # logging.info("\nTesting coordinate mapping...")

    # Create a simple curvilinear dataset similar to RASM

    # Simulate curvilinear coordinates like RASM
    ny, nx = 20, 30
    y = np.arange(ny)  # dimension coordinate
    x = np.arange(nx)  # dimension coordinate

    # Create 2D coordinate arrays (like real curvilinear grids)
    # These represent the actual lat/lon values at each grid point
    lat_2d = np.random.uniform(15, 75, (ny, nx))  # latitude values
    lon_2d = np.random.uniform(200, 330, (ny, nx))  # longitude values

    # Create test data - this needs to be a Dataset to match the RASM structure
    ds = xr.Dataset(
        {"test_var": (["y", "x"], np.random.random((ny, nx)))},
        coords={
            "y": (["y"], y),
            "x": (["x"], x),
            "lat": (["y", "x"], lat_2d, {"units": "degrees_north"}),
            "lon": (["y", "x"], lon_2d, {"units": "degrees_east"}),
        },
    )

    # Create target grid
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(16, 75, 2.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(200, 330, 2.0), {"units": "degrees_east"}),
        }
    )

    # logging.info(" Creating regridder with curvilinear source and rectilinear target...")

    # Debug: Check grid type detection
    # logging.info("  Debug: Checking grid types...")
    # source_type = _get_grid_type(ds)
    # target_type = _get_grid_type(ds_out)
    # logging.info("    Source grid type: %s", source_type)
    # logging.info("    Target grid type: %s", target_type)
    # logging.info("    Source coordinates: %s", list(ds.coords))
    # logging.info("    Target coordinates: %s", list(ds_out.coords))
    # logging.info("    Source dims: %s", list(ds.dims))
    # logging.info("    Target dims: %s", list(ds_out.dims))

    # regridder = ds.regrid.build_regridder(ds_out, method="linear")
    # logging.info("  ✓ Success: %s created", type(regridder).__name__)

    # Apply regridding using the accessor method (which handles validation properly)
    ds.regrid.linear(ds_out)
    # logging.info("  ✓ Success: Regridding completed")
    # logging.info("    Original shape: %s", ds['test_var'].shape)
    # logging.info("    Result shape: %s", result_ds['test_var'].shape)
    # logging.info("    Result coordinates: %s", list(result_ds.coords))


def test_backward_compatibility():
    """Test that existing rectilinear-to-rectilinear workflows still work."""
    # logging.info("\nTesting backward compatibility...")

    # Create standard rectilinear data
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # logging.info("  Testing rectilinear-to-rectilinear regridding...")
    regridder = source_data.regrid.build_regridder(target_grid, method="linear")
    # logging.info("  ✓ Success: %s created", type(regridder).__name__)

    regridder()
    # logging.info("  ✓ Success: Regridding completed")
    # logging.info("    Original shape: %s", source_data.shape)
    # logging.info("    Result shape: %s", result.shape)


if __name__ == "__main__":
    # logging.info("Running comprehensive RASM coordinate validation tests...\n")

    # Test the main RASM scenario
    success1 = test_rasm_coordinate_validation()

    # Test coordinate mapping
    success2 = test_coordinate_mapping()

    # Test backward compatibility
    success3 = test_backward_compatibility()

    # logging.info("\n%s", '='*60)
    # logging.info("SUMMARY:")
    # logging.info("  RASM test: %s", 'PASS' if success1 else 'FAIL')
    # logging.info("  Coordinate mapping test: %s", 'PASS' if success2 else 'FAIL')
    # logging.info("  Backward compatibility test: %s", 'PASS' if success3 else 'FAIL')
    # logging.info("%s", '='*60)

    if all([success1, success2, success3]):
        # logging.info("🎉 ALL TESTS PASSED! The fix is working correctly.")
        pass
    else:
        # logging.info("❌ Some tests failed. Please review the implementation.")
        pass
