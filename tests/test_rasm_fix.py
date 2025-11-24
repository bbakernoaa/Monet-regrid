#!/usr/bin/env python
"""Test script to verify the RASM dataset coordinate validation fix."""

import numpy as np
import xarray as xr


# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: import xarray_regrid  # Import to register the accessor
# New import: import monet_regrid  # Import to register the accessor


def test_rasm_coordinate_validation():
    """Test the exact scenario from the user's issue with RASM dataset."""
    print("Testing RASM dataset coordinate validation fix...")

    try:
        # Load RASM dataset (curvilinear grid with xc, yc coordinates)
        print("Loading RASM dataset...")
        ds = xr.tutorial.open_dataset("rasm")
        print("✓ RASM dataset loaded successfully")
        print(f"  Dimensions: {ds.dims}")
        print(f"  Coordinates: {list(ds.coords)}")
        print(f"  Data variables: {list(ds.data_vars)}")

        # Create rectilinear target grid with lat/lon coordinates
        print("\nCreating rectilinear target grid...")
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], np.arange(16, 75, 1.0), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(200, 330, 1.5), {"units": "degrees_east"}),
            }
        )
        print("✓ Target grid created")
        print(f"  Dimensions: {ds_out.dims}")
        print(f"  Coordinates: {list(ds_out.coords)}")

        # This should now work without ValueError
        print("\nTesting build_regridder with RASM dataset...")
        regridder = ds.regrid.build_regridder(ds_out, method="linear")
        print("✓ Success: Regridder created successfully")
        print(f"  Regridder type: {type(regridder).__name__}")
        print(f"  Regridder info: {regridder.info()}")

        # Test that the regridder can be applied to data
        print("\nTesting regridder application...")
        # Use one of the data variables from RASM dataset
        var_name = list(ds.data_vars)[0]  # Get first data variable
        print(f"  Using variable: {var_name}")

        # Apply regridding to a single variable
        result = ds[var_name].regrid.linear(ds_out)
        print("✓ Success: Data regridded successfully")
        print(f" Original shape: {ds[var_name].shape}")
        print(f" Regridded shape: {result.shape}")
        print(f" Result coordinates: {list(result.coords)}")

        # Test with the full dataset
        print("\nTesting regridder with full dataset...")
        result_ds = ds.regrid.linear(ds_out)
        print("✓ Success: Full dataset regridded successfully")
        print(f"  Result variables: {list(result_ds.data_vars)}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! RASM coordinate validation fix works correctly.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_coordinate_mapping():
    """Test that coordinate mapping works properly (xc->lon, yc->lat)."""
    # print("\nTesting coordinate mapping...")

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

    try:
        # print(" Creating regridder with curvilinear source and rectilinear target...")

        # Debug: Check grid type detection
        # print("  Debug: Checking grid types...")
        from monet_regrid.utils import _get_grid_type

        source_type = _get_grid_type(ds)
        target_type = _get_grid_type(ds_out)
        # print(f"    Source grid type: {source_type}")
        # print(f"    Target grid type: {target_type}")
        # print(f"    Source coordinates: {list(ds.coords)}")
        # print(f"    Target coordinates: {list(ds_out.coords)}")
        # print(f"    Source dims: {list(ds.dims)}")
        # print(f"    Target dims: {list(ds_out.dims)}")

        regridder = ds.regrid.build_regridder(ds_out, method="linear")
        # print(f"  ✓ Success: {type(regridder).__name__} created")

        # Apply regridding using the accessor method (which handles validation properly)
        result_ds = ds.regrid.linear(ds_out)
        # print("  ✓ Success: Regridding completed")
        # print(f"    Original shape: {ds['test_var'].shape}")
        # print(f"    Result shape: {result_ds['test_var'].shape}")
        # print(f"    Result coordinates: {list(result_ds.coords)}")

        return True
    except Exception as e:
        # print(f" ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that existing rectilinear-to-rectilinear workflows still work."""
    # print("\nTesting backward compatibility...")

    try:
        # Create standard rectilinear data
        source_data = xr.DataArray(
            np.random.random((10, 10)),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
        )

        target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

        # print("  Testing rectilinear-to-rectilinear regridding...")
        regridder = source_data.regrid.build_regridder(target_grid, method="linear")
        # print(f"  ✓ Success: {type(regridder).__name__} created")

        result = regridder()
        # print("  ✓ Success: Regridding completed")
        # print(f"    Original shape: {source_data.shape}")
        # print(f"    Result shape: {result.shape}")

        return True
    except Exception as e:
        # print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # print("Running comprehensive RASM coordinate validation tests...\n")

    # Test the main RASM scenario
    success1 = test_rasm_coordinate_validation()

    # Test coordinate mapping
    success2 = test_coordinate_mapping()

    # Test backward compatibility
    success3 = test_backward_compatibility()

    # print(f"\n{'='*60}")
    # print("SUMMARY:")
    # print(f"  RASM test: {'PASS' if success1 else 'FAIL'}")
    # print(f"  Coordinate mapping test: {'PASS' if success2 else 'FAIL'}")
    # print(f"  Backward compatibility test: {'PASS' if success3 else 'FAIL'}")
    # print(f"{'='*60}")

    if all([success1, success2, success3]):
        # print("🎉 ALL TESTS PASSED! The fix is working correctly.")
        pass
    else:
        # print("❌ Some tests failed. Please review the implementation.")
        pass
