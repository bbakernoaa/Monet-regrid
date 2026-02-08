# Curvilinear Grid Examples

"""
This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 [Your Organization].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications: Package renamed from xarray-regrid to monet-regrid,
URLs updated, and documentation adapted for new branding.
"""

This section provides examples demonstrating how to use monet-regrid with curvilinear grids.

## Example 1: Basic Curvilinear Regridding

This example shows how to regrid data from a curvilinear source grid to a rectilinear target grid using the `build_regridder` factory.

```python
import xarray as xr
import numpy as np

# Create a sample curvilinear source grid
source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
source_lon = -100 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

source_grid = xr.Dataset({
    'latitude': (['y', 'x'], source_lat),
    'longitude': (['y', 'x'], source_lon)
})

# Create sample data for the source grid
data_values = np.random.rand(6, 5)  # (y, x)
source_data = xr.DataArray(
    data_values,
    dims=['y', 'x'],
    coords={'y': range(6), 'x': range(5)}
)

# Create a rectilinear target grid
target_lat_rect = np.linspace(-5, 5, 3)
target_lon_rect = np.linspace(-5, 5, 4)
target_grid_rect = xr.Dataset({
    'latitude': (['y_target'], target_lat_rect),
    'longitude': (['x_target'], target_lon_rect)
})

# Build the regridder for curvilinear to rectilinear
regridder = source_data.regrid.build_regridder(target_grid_rect, method="linear")
regridded_data = regridder()

print("Regridded data (curvilinear to rectilinear):")
print(regridded_data)
```

## Example 2: Curvilinear to Curvilinear Regridding with Nearest Neighbor

This example demonstrates regridding between two curvilinear grids using the nearest neighbor method.

```python
import xarray as xr
import numpy as np

# Create a sample curvilinear source grid
source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
source_lat = 30 + 0.5 * source_x + 0.1 * source_y  # Curvilinear lat
source_lon = -100 + 0.3 * source_x + 0.2 * source_y  # Curvilinear lon

source_grid = xr.Dataset({
    'latitude': (['y', 'x'], source_lat),
    'longitude': (['y', 'x'], source_lon)
})

# Create sample data for the source grid
data_values = np.random.rand(6, 5)  # (y, x)
source_data = xr.DataArray(
    data_values,
    dims=['y', 'x'],
    coords={'y': range(6), 'x': range(5)}
)

# Create a second curvilinear target grid
target_x, target_y = np.meshgrid(np.linspace(0, 4, 3), np.linspace(0, 5, 4))
target_lat = 30 + 0.5 * target_x + 0.1 * target_y
target_lon = -100 + 0.3 * target_x + 0.2 * target_y

target_grid = xr.Dataset({
    'latitude': (['y_target', 'x_target'], target_lat),
    'longitude': (['y_target', 'x_target'], target_lon)
})

# Build the regridder for curvilinear to curvilinear
regridder = source_data.regrid.build_regridder(target_grid, method="nearest")
regridded_data = regridder()

print("\nRegridded data (curvilinear to curvilinear, nearest neighbor):")
print(regridded_data)
```

## Example 3: Handling Spherical Geometry and Fill Methods

This example shows how to use the `spherical` and `fill_method` options when regridding curvilinear grids.

```python
import xarray as xr
import numpy as np

# Create a sample curvilinear source grid
source_x, source_y = np.meshgrid(np.arange(5), np.arange(6))
source_lat = 30 + 0.5 * source_x + 0.1 * source_y
source_lon = -100 + 0.3 * source_x + 0.2 * source_y

source_grid = xr.Dataset({
    'latitude': (['y', 'x'], source_lat),
    'longitude': (['y', 'x'], source_lon)
})

# Create sample data
data_values = np.arange(30).reshape(6, 5)
source_data = xr.DataArray(
    data_values,
    dims=['y', 'x'],
    coords={'y': range(6), 'x': range(5)}
)

# Create a target grid that extends beyond the source grid
target_x, target_y = np.meshgrid(np.linspace(-1, 5, 4), np.linspace(-1, 6, 5))
target_lat = 30 + 0.5 * target_x + 0.1 * target_y
target_lon = -100 + 0.3 * target_x + 0.2 * target_y

target_grid = xr.Dataset({
    'latitude': (['y_target', 'x_target'], target_lat),
    'longitude': (['y_target', 'x_target'], target_lon)
})

# Build the regridder with spherical=True and fill_method='nearest'
# This will use 3D coordinate transformations and fill out-of-bounds areas
# with the nearest valid data point.
regridder = source_data.regrid.build_regridder(
    target_grid,
    method="linear",
    spherical=True,
    fill_method="nearest"
)
regridded_data = regridder()

print("\nRegridded data with spherical geometry and nearest fill method:")
print(regridded_data)

## Example 4: Statistical Regridding of Curvilinear Data

This example demonstrates how to use the `.regrid.stat()` method on curvilinear data to compute area-weighted averages on a rectilinear grid.

```python
import xarray as xr
import numpy as np

# Create a sample curvilinear source grid
source_x, source_y = np.meshgrid(np.arange(10), np.arange(10))
source_lat = 30 + 0.5 * source_x + 0.1 * source_y
source_lon = -100 + 0.3 * source_x + 0.2 * source_y

# Create sample data
data_values = np.ones((10, 10))
source_data = xr.Dataset(
    {"temperature": (("y", "x"), data_values)},
    coords={
        "lat": (("y", "x"), source_lat),
        "lon": (("y", "x"), source_lon),
    },
)

# Create a coarse rectilinear target grid
target_ds = xr.Dataset(
    coords={
        "lat": (("lat",), np.array([32.0, 35.0])),
        "lon": (("lon",), np.array([-98.0, -95.0])),
    }
)

# Perform statistical reduction (mean)
# This will automatically use flox to bin the curvilinear data into the target grid
regridded_ds = source_data.regrid.stat(target_ds, method="mean")

print("\nCoarse regridded statistical mean:")
print(regridded_ds.temperature)
```