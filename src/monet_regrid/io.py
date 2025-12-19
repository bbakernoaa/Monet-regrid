"""
I/O functions for monet-regrid.

This file is part of monet-regrid.

monet-regrid is a derivative work of xarray-regrid.
Original work Copyright (c) 2023-2025 Bart Schilperoort, Yang Liu.
This derivative work Copyright (c) 2025 monet-regrid Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law_ or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications: Package renamed from xarray-regrid to monet-regrid,
URLs updated, and documentation adapted for new branding.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from monet_regrid.core import BaseRegridder


def _regridder_to_netcdf(regridder: BaseRegridder, filepath: str) -> None:
    """Write regridder to a netCDF file."""
    regridder.source_data.to_netcdf(filepath, mode="w", group="source_data", engine="h5netcdf")
    regridder.target_grid.to_netcdf(filepath, mode="a", group="target_grid", engine="h5netcdf")

    regridder_config = xr.Dataset()
    info = regridder.info()
    info["method_kwargs"] = json.dumps(info["method_kwargs"])
    info["source_dims"] = json.dumps(info["source_dims"])
    info["target_coords"] = json.dumps(info["target_coords"])
    regridder_config.attrs.update(info)
    regridder_config.to_netcdf(filepath, mode="a", group="regridder_config", engine="h5netcdf")


def _regridder_from_netcdf(filepath: str) -> dict[str, Any]:
    """Read regridder from a netCDF file."""
    with xr.open_dataset(filepath, group="regridder_config", engine="h5netcdf") as ds:
        regridder_config = ds.attrs
        regridder_config["method_kwargs"] = json.loads(regridder_config["method_kwargs"])
        regridder_config["source_dims"] = json.loads(regridder_config["source_dims"])
        regridder_config["target_coords"] = json.loads(regridder_config["target_coords"])

    source_data = xr.open_dataset(filepath, group="source_data", engine="h5netcdf")
    target_grid = xr.open_dataset(filepath, group="target_grid", engine="h5netcdf")

    regridder_config.update(
        {
            "source_data": source_data,
            "target_grid": target_grid,
        }
    )

    return regridder_config
