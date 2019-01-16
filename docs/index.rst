Welcome to the terrainbento documentation!
==============================================

terrainbento is a python package for multi-model analysis in Earth surface
dynamics. It was built on top of the `Landlab Toolkit`_.

.. _Landlab Toolkit: https://landlab.github.io

You can find all the source code, request additional features, or report a bug
in the `GitHub repository`_.

.. _GitHub repository: https://github.com/TerrainBento/terrainbento

The goal of terrainbento is to make it easier to create alternative Earth
surface dynamics models. The package has six main parts that support this
goal. First, a model base class called
:py:class:`~terrainbento.base_class.erosion_model.ErosionModel` contains the
functionality requires across models (e.g. reading in input files, writing
output). Two specialized base classes also exist for models that use stochastic
hydrology
(:py:class:`~terrainbento.base_class.stochastic_erosion_model.StochasticErosionModel`)
or multiple lithology layers
(:py:class:`~terrainbento.base_class.two_lithology_erosion_model.TwoLithologyErosionModel`)
.

Second, a set of :py:mod:`Boundary Condition Handlers <terrainbento.boundary_handlers>`
provide tools to set and modify boundary conditions during a model run.

Third, a set of :py:mod`Precipitators <terrainbento.precipitators>` permits
alternative approaches to specifying spatially and temporally variable
precipitation.

Fourth, a set of :py:mod:`Runoff Generators <terrainbento.runoff_generators>`
allows for alternative approaches for converting rainfall into runoff.

Fifth, we provide an initial set of landscape evolution models derived from the
base class. These models increase in complexity from a base model to models
with one, two, or three differences from the base mode.

Finally, a
:py:class:`~terrainbento.model_template.model_template.ModelTemplate` provides an skeleton of
a model made with the main base class that can be used to create your own
terrainbento model.

A number of Jupyter Notebook Tutorials have been developed to highlight how to
use terrainbento. They can be found in `the terrainbento tutorials repository`_.

.. _the terrainbento tutorials repository: https://github.com/TerrainBento/examples_tests_and_tutorials


Model Base Class
----------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.base_class

Clock
-----

.. toctree::
   :maxdepth: 2

   source/terrainbento.clock

Boundary Condition Handlers
---------------------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.boundary_handlers

Precipitators
-------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.precipitators


RunoffGenerators
----------------

.. toctree::
   :maxdepth: 2

   source/terrainbento.runoff_generators

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
