"""
Tests for package documentation and docstrings.
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
"""
from monet_regrid.core import CurvilinearRegridder, RectilinearRegridder


def test_regridder_init_docstrings():
    """Verify that regridder __init__ methods have NumPy-style docstrings."""
    rectilinear_doc = RectilinearRegridder.__init__.__doc__
    curvilinear_doc = CurvilinearRegridder.__init__.__doc__

    assert rectilinear_doc is not None, "RectilinearRegridder.__init__ is missing a docstring."
    assert "Parameters" in rectilinear_doc, "RectilinearRegridder.__init__ docstring is missing a 'Parameters' section."

    assert curvilinear_doc is not None, "CurvilinearRegridder.__init__ is missing a docstring."
    assert "Parameters" in curvilinear_doc, "CurvilinearRegridder.__init__ docstring is missing a 'Parameters' section."
