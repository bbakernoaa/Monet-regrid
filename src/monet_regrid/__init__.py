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

from monet_regrid import methods
from monet_regrid.accessor import Regridder
from monet_regrid.constants import GridType
from monet_regrid.core import BaseRegridder, CurvilinearRegridder, RectilinearRegridder
from monet_regrid.utils import Grid, create_regridding_dataset

__all__ = [
    "BaseRegridder",
    "CurvilinearRegridder",
    "Grid",
    "GridType",
    "RectilinearRegridder",
    "Regridder",
    "create_regridding_dataset",
    "methods",
]

__version__ = "0.5.0"
