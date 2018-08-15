[![Build Status](https://travis-ci.org/TerrainBento/terrainbento.svg?branch=master)](https://travis-ci.org/TerrainBento/terrainbento)
[![Build status](https://ci.appveyor.com/api/projects/status/kwwpjifg8vrwe51x/branch/master?svg=true)](https://ci.appveyor.com/project/kbarnhart/terrainbento/branch/master)
[![Anaconda-Server Badge](https://anaconda.org/terrainbento/terrainbento/badges/version.svg)](https://anaconda.org/terrainbento/terrainbento)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![Coverage Status](https://coveralls.io/repos/github/TerrainBento/terrainbento/badge.svg?branch=master)](https://coveralls.io/github/TerrainBento/terrainbento?branch=master)
[![Documentation Status](https://readthedocs.org/projects/terrainbento/badge/?version=latest)](http://terrainbento.readthedocs.io/en/latest/?badge=latest)
[![Code Health](https://landscape.io/github/TerrainBento/terrainbento/master/landscape.svg?style=flat)](https://landscape.io/github/TerrainBento/terrainbento/master) 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7fcb775a6c3044cda4429ed1c1dac2e8)](https://www.codacy.com/app/katy.barnhart/terrainbento?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=TerrainBento/terrainbento&amp;utm_campaign=Badge_Grade)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# terrainbento

Currently in development.

A modular landscape evolution modeling package built on top of the [Landlab Toolkit](http://landlab.github.io).

terrainbento's User Manual is located at our [Read The Docs page](http://terrainbento.readthedocs.io/).

We recommend that you start with [this set of Jupyter notebooks](https://github.com/TerrainBento/examples_tests_and_tutorials) that introduce terrainbento .

A manuscript describing terrainbento can be found [here]() ** not yet submitted, link does not exist **.

## A quick example

The following is the code needed to run the Basic model and create the gif below.

```python
from terrainbento import Basic

model = Basic(params = {'dt':100, 
                        'run_duration':100000, 
                        'output_interval':1000, 
                        'm_sp': 0.5, 
                        'n_sp': 1, 
                        'water_erodability':0.001, 
                        'regolith_transport_parameter': 0.1, 
                        'number_of_node_rows': 50, 
                        'number_of_node_columns':50,
                        'node_spacing': 10,
                        'initial_elevation': 1000, 
                        'add_initial_elevation_to_all_nodes': False})
model.run()
ds = model.to_xarray_dataset()


```

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
