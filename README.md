# monet-regrid: Regridding utilities for xarray.

With monet-regrid it is possible to regrid between two grids. Both Rectilinear and Curvilinear grids are supported. The following methods are supported:
 - Linear (Rectilinear and Curvilinear) 
 - Nearest-neighbor (Rectilinear and Curvilinear) 
 - Conservative (Rectilinear and Curvilinear)
 - Cubic (Rectilinear and Curvilinear)
 - Bilinear (Rectilinear and Curvilinear)
 - "Most common value", as well as other zonal statistics (e.g., variance or median) (Rectilinear and Curvilinear)

All regridding methods can operate lazily on [Dask arrays](https://docs.xarray.dev/en/latest/user-guide/dask.html).

Note that "Most common value" is designed to regrid categorical data to a coarse resolution. For regridding categorical data to a finer resolution, please use "nearest-neighbor" regridder.

[![PyPI](https://img.shields.io/pypi/v/monet-regrid.svg?style=flat)](https://pypi.python.org/pypi/monet-regrid/)
[![conda-forge](https://anaconda.org/conda-forge/monet-regrid/badges/version.svg)](https://anaconda.org/conda-forge/monet-regrid)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10203304.svg)](https://doi.org/10.5281/zenodo.10203304)
[![Docs](https://readthedocs.org/projects/monet-regrid/badge/?version=latest&style=flat)](https://monet-regrid.readthedocs.org/)

## Why monet-regrid?

Regridding is a common operation in earth science and other fields. While xarray does have some interpolation methods available, these are not always straightforward to use. Additionally, methods such as conservative regridding, or taking the most common value, are not available in xarray.

## Installation

For a minimal install:
```console
pip install monet-regrid
```

*Note: monet-regrid is also [available on conda-forge](https://anaconda.org/conda-forge/monet-regrid).*

To improve performance in certain cases:
```console
pip install monet-regrid[accel]
```

which includes optional extras such as:
 - `dask`: parallelization over chunked data
 - `sparse`: for performing conservative regridding using sparse weight matrices
 - `opt-einsum`: optimized einsum routines used in conservative regridding

Benchmarking varies across different hardware specifications, but the inclusion of these extras can often provide significant speedups.

## Usage
The monet-regrid routines are accessed using the "regrid" accessor on an xarray Dataset:
```py
import monet_regrid

ds = xr.open_dataset("input_data.nc")
ds_grid = xr.open_dataset("target_grid.nc")

ds.regrid.linear(ds_grid)
```

For examples, see the benchmark notebooks and the demo notebooks.

## Benchmarks
The benchmark notebooks contain comparisons to more standard methods (CDO, xESMF).

To be able to run the notebooks, a conda environment is required (due to ESMF and CDO).
You can install this environment using the `environment.yml` file in this repository.
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a lightweight version of the much faster "mamba" conda alternative.

```sh
micromamba create -n environment_name -f environment.yml
```

## Acknowledgements

monet-regrid was adapted from [xarray-regrid](https://xarray-regrid.readthedocs.io/en/latest/)

**Dependencies:**
*   `numpy`
*   `xarray`
*   `flox`
*   `scipy`
*   `pyproj`
*   `cf-xarray`
