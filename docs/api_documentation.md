# API Documentation

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

This section provides detailed documentation for the core regridder classes and the `build_regridder` factory method.

## `BaseRegridder`

An abstract base class for all regridder implementations. It defines the common interface and ensures a consistent API across different grid types.

### Methods

*   `__call__(self, **kwargs)`: Executes the regridding operation.
*   `to_file(self, filepath: str, **kwargs)`: Saves the regridder configuration to a file.
*   `from_file(cls, filepath: str, **kwargs)`: Loads a regridder from a file.
*   `info(self)`: Returns a dictionary containing regridder metadata and configuration.

## `RectilinearRegridder`

Handles regridding between rectilinear grids using various interpolation methods.

### Initialization

```python
RectilinearRegridder(
    source_data: xr.DataArray | xr.Dataset,
    target_grid: xr.Dataset,
    method: str = "linear",
    time_dim: str | None = "time",
    **kwargs
)
```

*   `source_data`: The source data (DataArray or Dataset).
*   `target_grid`: The target grid specification (Dataset).
*   `method`: Interpolation method ('linear', 'nearest', 'cubic', 'conservative').
*   `time_dim`: Name of the time dimension.
*   `**kwargs`: Additional method-specific arguments.

### Methods

*   `__call__(self, **kwargs)`: Executes the regridding operation.
*   `to_file(self, filepath: str, **kwargs)`: Saves the regridder configuration.
*   `from_file(cls, filepath: str, **kwargs)`: Loads a regridder from a file.
*   `info(self)`: Returns regridder information.
*   `stat(self, method: str, time_dim: str | None = "time", skipna: bool = False, fill_value: Any | None = None)`: Upsamples data using statistical methods.
*   `most_common(self, values: np.ndarray, time_dim: str | None = "time", fill_value: Any | None = None)`: Regrids by taking the most common value.
*   `least_common(self, values: np.ndarray, time_dim: str | None = "time", fill_value: Any | None = None)`: Regrids by taking the least common value.

## `CurvilinearRegridder`

Handles regridding between curvilinear grids using 3D coordinate transformations.

### Initialization

```python
CurvilinearRegridder(
    source_data: xr.DataArray | xr.Dataset,
    target_grid: xr.Dataset,
    method: str = "linear",
    **kwargs
)
```

*   `source_data`: The source data (DataArray or Dataset).
*   `target_grid`: The target grid specification (Dataset).
*   `method`: Interpolation method for curvilinear grids.
*   `**kwargs`: Additional method-specific arguments.

### Configuration Options

*   `spherical` (bool): Whether to use spherical barycentrics (default: `True`).
*   `fill_method` (Literal["nan", "nearest"]): How to handle out-of-domain targets (default: `"nan"`).
*   `extrapolate` (bool): Whether to allow extrapolation beyond the source domain (default: `False`).

### Methods

*   `__call__(self, **kwargs)`: Executes the regridding operation for curvilinear grids.
*   `to_file(self, filepath: str, **kwargs)`: Saves the regridder configuration.
*   `from_file(cls, filepath: str, **kwargs)`: Loads a regridder from a file.
*   `info(self)`: Returns regridder information.
*   `stat(self, method: str, time_dim: str | None = "time", skipna: bool = False, fill_value: Any | None = None)`: Upsamples curvilinear data using statistical methods (e.g., mean, sum).
*   `most_common(self, values: np.ndarray, time_dim: str | None = "time", fill_value: Any | None = None)`: Regrids curvilinear categorical data by taking the most common value.
*   `least_common(self, values: np.ndarray, time_dim: str | None = "time", fill_value: Any | None = None)`: Regrids curvilinear categorical data by taking the least common value.

## `build_regridder` Factory Method

This factory method, accessed via the `.regrid` accessor, automatically detects grid types and dispatches to the appropriate regridder.

### Usage

```python
regridder = source_data.regrid.build_regridder(target_grid, method="linear")
result = regridder()
```

This method simplifies the regridding process by abstracting away the need to manually select between `RectilinearRegridder` and `CurvilinearRegridder`. It supports all available regridding methods and can pass through additional arguments to the underlying regridder implementations.
