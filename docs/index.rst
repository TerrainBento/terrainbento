.. terrainbento documentation master file, created by
   sphinx-quickstart on Tue Mar  6 11:22:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the terrainbento documentation!
==============================================

terrainbento is a python package for multi-model analysis in Earth surface
dynamics. It was built on top of the `Landlab Toolkit`_.

.. _Landlab Toolkit: https://landlab.github.io

You can find all the source code, request additional features, or report a bug
in the `GitHub repository`_.

.. _GitHub repository: https://github.com/TerrainBento/terrainbento

The goal of terrainbento is to make it easier to create alternative Earth
surface dynamics models. The package has four main parts that support this
goal. First, a model base class called **ErosionModel** contains the
functionality requires across models (e.g. reading in input files, writing out
output). Two specialized base classes also exist for models that use stochastic
hydrology or multiple lithology layers.

Second, a set of **Boundary Condition Handlers** provide tools to set and modify
boundary conditions during a model run. Third, we provide an initial set of
landscape evolution models derived from the base class. These models increase
in complexity from a base model to models with one, two, or three differences
from the base mode.

Finally, a **ModelTemplate** provides an skeleton of a model made with the main
base class that can be used to create your own terrainbento model.

A number of Jupyter Notebook Tutorials have been developed to highlight how to
use terrainbento. They can be found in `the terrainbento tutorials repository`_.

.. _the terrainbento tutorials repository: https://github.com/TerrainBento/examples_tests_and_tutorials


Model Base Class
----------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.base_class

Boundary Condition Handlers
---------------------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.boundary_handlers

Derived Models
--------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.derived_models

Model Template
--------------

 .. toctree::
    :maxdepth: 2

    source/terrainbento.model_template

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
