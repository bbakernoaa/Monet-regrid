"""Tests for the build_regridder factory API with backward compatibility."""

import numpy as np
import pytest
import xarray as xr

import monet_regrid as xrg

# REBRAND NOTICE: This test file has been updated to use the new monet_regrid package.
# Old import: import monet_regrid as xrg
# New import: import monet_regrid as xrg


def test_build_regridder_factory_method():
    """Test the build_regridder factory method directly."""
    # Create sample data
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    # Test the build_regridder method
    regridder_accessor = xrg.Regridder(source_data)
    built_regridder = regridder_accessor.build_regridder(target_grid, method="linear")

    # Verify the correct type is returned
    assert isinstance(built_regridder, xrg.RectilinearRegridder)
    assert built_regridder.method == "linear"

    # Verify it works correctly
    result = built_regridder()
    assert "lat" in result.coords
    assert "lon" in result.coords
    assert result.shape == (8, 8)


def test_build_regridder_with_different_methods():
    """Test the build_regridder method with different regridding methods."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder_accessor = xrg.Regridder(source_data)

    # Test different methods
    for method in ["linear", "nearest", "cubic"]:
        built_regridder = regridder_accessor.build_regridder(target_grid, method=method)
        assert isinstance(built_regridder, xrg.RectilinearRegridder)
        assert built_regridder.method == method

        result = built_regridder()
        assert result.shape == (8, 8)


def test_build_regridder_with_conservative_method():
    """Test the build_regridder method with conservative method."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
        attrs={"units": "degrees"},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder_accessor = xrg.Regridder(source_data)
    built_regridder = regridder_accessor.build_regridder(
        target_grid, method="conservative", skipna=True, nan_threshold=0.5
    )

    assert isinstance(built_regridder, xrg.RectilinearRegridder)
    assert built_regridder.method == "conservative"

    result = built_regridder()
    assert result.shape == (8, 8)


def test_backward_compatibility_linear():
    """Test that the linear method still works the same way."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder = xrg.Regridder(source_data)

    # Original way should still work
    result1 = regridder.linear(target_grid)

    # New way should produce the same result
    built_regridder = regridder.build_regridder(target_grid, method="linear")
    result2 = built_regridder()

    # Results should have the same shape and be close in values
    assert result1.shape == result2.shape
    np.testing.assert_allclose(result1.values, result2.values, rtol=1e-10)


def test_backward_compatibility_nearest():
    """Test that the nearest method still works the same way."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder = xrg.Regridder(source_data)

    # Original way should still work
    result1 = regridder.nearest(target_grid)

    # New way should produce the same result
    built_regridder = regridder.build_regridder(target_grid, method="nearest")
    result2 = built_regridder()

    # Results should have the same shape and be close in values
    assert result1.shape == result2.shape
    np.testing.assert_allclose(result1.values, result2.values, rtol=1e-10)


def test_backward_compatibility_cubic():
    """Test that the cubic method still works the same way."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder = xrg.Regridder(source_data)

    # Original way should still work
    result1 = regridder.cubic(target_grid)

    # New way should produce the same result
    built_regridder = regridder.build_regridder(target_grid, method="cubic")
    result2 = built_regridder()

    # Results should have the same shape and be close in values
    assert result1.shape == result2.shape
    np.testing.assert_allclose(result1.values, result2.values, rtol=1e-10)


def test_backward_compatibility_conservative():
    """Test that the conservative method still works the same way."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
        attrs={"units": "degrees"},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder = xrg.Regridder(source_data)

    # Original way should still work
    result1 = regridder.conservative(target_grid)

    # New way should produce the same result
    built_regridder = regridder.build_regridder(target_grid, method="conservative")
    result2 = built_regridder()

    # Results should have the same shape and be close in values
    assert result1.shape == result2.shape
    np.testing.assert_allclose(result1.values, result2.values, rtol=1e-10)


def test_build_regridder_with_dataset():
    """Test the build_regridder method with Dataset input."""
    source_data = xr.Dataset(
        {"var1": (["lat", "lon"], np.random.random((10, 10))), "var2": (["lat", "lon"], np.random.random((10, 10)))},
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder_accessor = xrg.Regridder(source_data)
    built_regridder = regridder_accessor.build_regridder(target_grid, method="linear")

    assert isinstance(built_regridder, xrg.RectilinearRegridder)
    assert built_regridder.method == "linear"

    result = built_regridder()
    assert "var1" in result.data_vars
    assert "var2" in result.data_vars
    assert result["var1"].shape == (8, 8)


def test_build_regridder_method_parameters():
    """Test that additional parameters are passed correctly."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
        attrs={"units": "degrees"},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder_accessor = xrg.Regridder(source_data)

    # Test with additional parameters
    built_regridder = regridder_accessor.build_regridder(
        target_grid, method="conservative", skipna=False, nan_threshold=0.8, latitude_coord="lat"
    )

    assert isinstance(built_regridder, xrg.RectilinearRegridder)
    assert built_regridder.method == "conservative"
    # Verify that parameters were passed through
    assert built_regridder.method_kwargs["skipna"] is False
    assert built_regridder.method_kwargs["nan_threshold"] == 0.8
    assert built_regridder.method_kwargs["latitude_coord"] == "lat"


def test_build_regridder_default_method():
    """Test that the build_regridder method uses the correct default."""
    source_data = xr.DataArray(
        np.random.random((10, 10)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, 10), "lon": np.linspace(-5, 5, 10)},
    )

    target_grid = xr.Dataset({"lat": ("lat", np.linspace(-4, 4, 8)), "lon": ("lon", np.linspace(-4, 4, 8))})

    regridder_accessor = xrg.Regridder(source_data)

    # Test with default method
    built_regridder = regridder_accessor.build_regridder(target_grid)

    assert isinstance(built_regridder, xrg.RectilinearRegridder)
    assert built_regridder.method == "linear"  # Default method


if __name__ == "__main__":
    pytest.main([__file__])
