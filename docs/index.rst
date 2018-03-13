.. terrainbento documentation master file, created by
   sphinx-quickstart on Tue Mar  6 11:22:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ``terrainbento`` documentation!
========================================

``terrainbento`` is a python package for multi-model analysis in Earth surface
dynamics. It was built on top of the `Landlab Toolkit`_.

.. _Landlab Toolkit: https://landlab.github.io

The goal of ``terrainbento`` is to make it easier to create alternative Earth
surface dynamics models. The package has three main parts that support this
goal. First, a **Model Base Class** that contains the functionality required
across models (e.g. reading in input files, writing out output). Second, a set
of **Boundary Condition Handlers** provide tools to set and modify boundary
conditions during a model run. Third, we provide an initial set of landscape
evolution models derived from the base class.


Model Base Class
------------------------------------------

.. toctree::
   :maxdepth: 4

   source/terrainbento.base_class

Boundary Condition Handlers
------------------------------------------
 .. toctree::
    :maxdepth: 4

    source/terrainbento.boundary_condition_handlers

Derived Models
------------------------------------------
 .. toctree::
    :maxdepth: 4
    source/terrainbento.derived_models


Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
