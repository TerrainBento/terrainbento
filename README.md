[![Build Status](https://travis-ci.org/TerrainBento/terrainbento.svg?branch=master)](https://travis-ci.org/TerrainBento/terrainbento)
[![Build status](https://ci.appveyor.com/api/projects/status/kwwpjifg8vrwe51x/branch/master?svg=true)](https://ci.appveyor.com/project/kbarnhart/terrainbento/branch/master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/123941145.svg)](https://zenodo.org/badge/latestdoi/123941145)

[![Coverage Status](https://coveralls.io/repos/github/TerrainBento/terrainbento/badge.svg?branch=master)](https://coveralls.io/github/TerrainBento/terrainbento?branch=master)
[![Documentation Status](https://readthedocs.org/projects/terrainbento/badge/?version=latest)](http://terrainbento.readthedocs.io/en/latest/?badge=latest)
[![Code Health](https://landscape.io/github/TerrainBento/terrainbento/master/landscape.svg?style=flat)](https://landscape.io/github/TerrainBento/terrainbento/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7fcb775a6c3044cda4429ed1c1dac2e8)](https://www.codacy.com/app/katy.barnhart/terrainbento?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=TerrainBento/terrainbento&amp;utm_campaign=Badge_Grade)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# terrainbento

A modular landscape evolution modeling package built on top of the [Landlab Toolkit](http://landlab.github.io).

terrainbento"s User Manual is located at our [Read The Docs page](http://terrainbento.readthedocs.io/).

We recommend that you start with [this set of Jupyter notebooks](https://github.com/TerrainBento/examples_tests_and_tutorials) that introduce terrainbento .

A manuscript describing terrainbento is currently in discussion at GMD. The discussion paper can be found [here](https://www.geosci-model-dev-discuss.net/gmd-2018-204/).

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

    # Create the Grid.
    "grid": {"grid": {"RasterModelGrid":[(200, 320), {"xy_spacing": 10}]},
             "fields": {"at_node": {"topographic__elevation":{"random":[{"where":"CORE_NODE"}]}}}},

    # Set up Boundary Handlers
    "boundary_handlers":{"NotCoreNodeBaselevelHandler": {"modify_core_nodes": True,
                                                         "lowering_rate": -0.001}},
    # Parameters that control output.
    "output_interval": 1e3,
    "save_first_timestep": True,
    "fields":["topographic__elevation"],

    # Parameters that control process and rates.
    "water_erodability" : 0.001,
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

Before installing terrainbento you will need a python distribution. We recommend that you use the [Anaconda python distribution](https://www.anaconda.com/download/). Unless you have a specific reason to want Python 2.7 we strongly suggest that you install Python 3.6 (or the current 3.* version provided by Anaconda).

### Using conda
To install the release version of terrainbento (this is probably what you want) open a terminal and execute the following:

```
conda config --add channels conda-forge
conda install terrainbento
```

### From source code

To install the terrainbento source code version of terrainbento do the following:

#### Option A: You already have landlab installed (either through conda or through the source code)

```
git clone https://github.com/TerrainBento/terrainbento.git
cd terrainbento
conda install --file=requirements.txt
python setup.py install
```

#### Option B: You do not have landlab installed

```
conda config --add channels conda-forge
conda install landlab
git clone https://github.com/TerrainBento/terrainbento.git
cd terrainbento
conda install --file=requirements.txt
python setup.py install
```

#### A note to developers

If you plan to develop with terrainbento, please fork terrainbento, clone the forked repository, and replace `python setup.py install` with `python setup.py develop`. If you have any questions, please contact us by making an Issue.


## How to cite

A manuscript describing terrainbento is currently in discussion at GMD. The discussion paper can be found [here](https://www.geosci-model-dev-discuss.net/gmd-2018-204/).
