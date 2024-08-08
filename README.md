| Thing | Badge |
| :---: | :---: |
| CI Status | [![Test](https://github.com/TerrainBento/terrainbento/actions/workflows/test.yml/badge.svg)](https://github.com/TerrainBento/terrainbento/actions/workflows/test.yml) |
| Coverage | [![Coverage Status](https://coveralls.io/repos/github/TerrainBento/terrainbento/badge.svg?branch=master)](https://coveralls.io/github/TerrainBento/terrainbento?branch=master) |
| Docs | [![Documentation Status](https://readthedocs.org/projects/terrainbento/badge/?version=latest)](http://terrainbento.readthedocs.io/en/latest/?badge=latest) |
| Notebooks | [![Notebooks](https://github.com/TerrainBento/terrainbento/actions/workflows/test-notebooks.yml/badge.svg)](https://github.com/TerrainBento/terrainbento/actions/workflows/test-notebooks.yml) |
| License | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |
| Style | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) |
| DOI | [![DOI](https://zenodo.org/badge/123941145.svg)](https://zenodo.org/badge/latestdoi/123941145) |
| Conda Recipe | [![Conda Recipe](https://img.shields.io/badge/recipe-terrainbento-green.svg)](https://anaconda.org/conda-forge/terrainbento) |
| Downloads | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/terrainbento.svg)](https://anaconda.org/conda-forge/terrainbento) |
| Version | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/terrainbento.svg)](https://anaconda.org/conda-forge/terrainbento) |
| Platforms | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/terrainbento.svg)](https://anaconda.org/conda-forge/terrainbento) |
| Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TerrainBento/terrainbento/master?filepath=notebooks%2FWelcome_to_TerrainBento.ipynb) |

# terrainbento

A modular landscape evolution modeling package built on top of the [Landlab Toolkit](http://landlab.github.io).

terrainbento"s User Manual is located at our [Read The Docs page](http://terrainbento.readthedocs.io/).

We recommend that you start with a set of Jupyter notebooks [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TerrainBento/terrainbento/master?filepath=notebooks%2FWelcome_to_TerrainBento.ipynb) that introduce terrainbento and the model description paper [Barnhart et al. (2019)](https://doi.org/10.5194/gmd-12-1267-2019). The link above goes to a Binder instance, if you want the notebooks themselves clone the repo and navigate to the directory `notebooks`.


## A quick example

The following is all the code needed to run the Basic model. There are a few
different options available to create a model, here we will create one from a
file-like object. The string will contain the same information as a YAML style
input file that specifies the model construction and run.

```python
from terrainbento import Basic

params = {
    # create the Clock.
    "clock": {"start": 0,
              "step": 10,
              "stop": 1e5},

    # Create the Grid
    "grid": {
        "RasterModelGrid": [
            (200, 320),
            {
                "xy_spacing": 10
            },
            {
                "fields": {
                    "node": {
                        "topographic__elevation": {
                            "random": [{
                                "where": "CORE_NODE"
                            }]
                        }
                    }
                }
            },
        ]
    },

    # Set up Boundary Handlers
    "boundary_handlers":{"NotCoreNodeBaselevelHandler": {"modify_core_nodes": True,
                                                         "lowering_rate": -0.001}},
    # Parameters that control output.
    "output_interval": 1e3,
    "save_first_timestep": True,
    "fields":["topographic__elevation"],

    # Parameters that control process and rates.
    "water_erodibility" : 0.001,
    "m_sp" : 0.5,
    "n_sp" : 1.0,
    "regolith_transport_parameter" : 0.2,
         }

model = Basic.from_dict(params)
model.run()
```

Next we make an image for each output interval.

```python
from landlab import imshow_grid

filenames = []
ds = model.to_xarray_dataset()
for i in range(ds.topographic__elevation.shape[0]):
    filename = "temp_output."+str(i)+".png"
    imshow_grid(model.grid, ds.topographic__elevation.values[i, :, :], cmap="viridis", limits=(0, 12), output=filename)
    filenames.append(filename)
model.remove_output_netcdfs()

```

Finally we compile the images into a gif.

```python
import os
import imageio
with imageio.get_writer("terrainbento_example.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        os.remove(filename)
```

![Example terrainbento run](https://github.com/TerrainBento/terrainbento/blob/master/docs/images/terrainbento_example.gif)

## Installation instructions

Before installing terrainbento you will need a Python distribution. We recommend that you use the [Anaconda python distribution](https://www.anaconda.com/download/).
We strongly suggest that you install the current 3.* version of Python.

To install the release version of terrainbento (this is probably what you want) we support conda and pip package management.

### Using conda
Open a terminal and execute the following:

```
conda config --add channels conda-forge
conda install terrainbento
```

### Using pip
Open a terminal and execute the following:

```
pip install terrainbento
```

### From source code

To install the terrainbento source code version of terrainbento we recommend creating a conda environment for terrainbento.

```
git clone https://github.com/TerrainBento/terrainbento.git
cd terrainbento
conda env create -f environment-dev.yml
conda activate terrainbento_dev
pip install .
```

#### Notes for developers

If you plan to develop with terrainbento, please fork terrainbento, clone the forked repository, and create an editable install with:
```bash
pip install -e .
```

We use [nox](https://nox.thea.codes/en/stable/) for most project tasks in terrainbento.
Install nox and list the available sessions with:
```bash
pip install -e ".[dev]"
nox --list
```
To use nox to run the terrainbento tests, for example, call the _test_ session:
```bash
nox -s test
```

If you have any questions, please contact us by making an [issue](https://github.com/TerrainBento/terrainbento/issues).


## How to cite

Barnhart, K. R., Glade, R. C., Shobe, C. M., and Tucker, G. E.: Terrainbento 1.0: a Python package for multi-model analysis in long-term drainage basin evolution, Geosci. Model Dev., 12, 1267-1297, https://doi.org/10.5194/gmd-12-1267-2019, 2019.
