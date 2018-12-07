[![Build Status](https://travis-ci.org/TerrainBento/terrainbento.svg?branch=master)](https://travis-ci.org/TerrainBento/terrainbento)
[![Build status](https://ci.appveyor.com/api/projects/status/kwwpjifg8vrwe51x/branch/master?svg=true)](https://ci.appveyor.com/project/kbarnhart/terrainbento/branch/master)
[![Anaconda-Server Badge](https://anaconda.org/terrainbento/terrainbento/badges/version.svg)](https://anaconda.org/terrainbento/terrainbento)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/123941145.svg)](https://zenodo.org/badge/latestdoi/123941145)

[![Coverage Status](https://coveralls.io/repos/github/TerrainBento/terrainbento/badge.svg?branch=master)](https://coveralls.io/github/TerrainBento/terrainbento?branch=master)
[![Documentation Status](https://readthedocs.org/projects/terrainbento/badge/?version=latest)](http://terrainbento.readthedocs.io/en/latest/?badge=latest)
[![Code Health](https://landscape.io/github/TerrainBento/terrainbento/master/landscape.svg?style=flat)](https://landscape.io/github/TerrainBento/terrainbento/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7fcb775a6c3044cda4429ed1c1dac2e8)](https://www.codacy.com/app/katy.barnhart/terrainbento?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=TerrainBento/terrainbento&amp;utm_campaign=Badge_Grade)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# terrainbento

Currently in development.

A modular landscape evolution modeling package built on top of the [Landlab Toolkit](http://landlab.github.io).

terrainbento"s User Manual is located at our [Read The Docs page](http://terrainbento.readthedocs.io/).

We recommend that you start with [this set of Jupyter notebooks](https://github.com/TerrainBento/examples_tests_and_tutorials) that introduce terrainbento .

A manuscript describing terrainbento is currently in discussion at GMD. The discussion paper can be found [here](https://www.geosci-model-dev-discuss.net/gmd-2018-204/). Please consider contributing comments or feedback as part of the review process.

## A quick example

The following is the code needed to run the Basic model and compile it into a
gif.

```python
from terrainbento import Basic

model = Basic(params={"dt" : 100,
                      "output_interval": 1e3,
                      "run_duration": 1.5e5,
                      "number_of_node_rows" : 200,
                      "number_of_node_columns" : 320,
                      "node_spacing" : 10.0,
                      "add_random_noise": True,
                      "initial_noise_std": 1.,
                      "random_seed": 4897,
                      "water_erodability" : 0.001,
                      "m_sp" : 0.5,
                      "n_sp" : 1.0,
                      "regolith_transport_parameter" : 0.2,
                      "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
                      "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True,
                                                      "lowering_rate": -0.001}})
model.run(output_fields='topographic__elevation')
model.to_gif("terrainbento_example.gif",
             field="topographic__elevation",
             limits = (0, 12))
```

![Example terrainbento run](https://github.com/TerrainBento/terrainbento/blob/master/docs/images/terrainbento_example.gif)

## Installation instructions

Before installing terrainbento you will need a python distribution. We recommend that you use the [Anaconda python distribution](https://www.anaconda.com/download/). Unless you have a specific reason to want Python 2.7 we strongly suggest that you install Python 3.6 (or the current 3.* version provided by Anaconda).

### Using conda
To install the release version of terrainbento (this is probably what you want) open a terminal and execute the following:

```
conda config --add channels landlab
conda install -c terrainbento terrainbento
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
conda install -c landlab landlab
git clone https://github.com/TerrainBento/terrainbento.git
cd terrainbento
conda install --file=requirements.txt
python setup.py install
```

#### A note to developers

If you plan to develop with terrainbento, please fork terrainbento, clone the forked repository, and replace `python setup.py install` with `python setup.py develop`.


## How to cite

There will be a GMD paper.
