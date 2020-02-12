.. py:module:: terrainbento.boundary_handlers

terrainbento Boundary Condition Handlers
========================================

Presently terrainbento has four built-in boundary condition handlers. In
addition, a small number of Landlab components are valid.


Time Varying Precipitation
--------------------------

.. toctree::

    terrainbento.boundary_handlers.precip_changer


Domain Boundary Elevation Modifiers
-----------------------------------

.. toctree::

    terrainbento.boundary_handlers.generic_function_baselevel_handler
    terrainbento.boundary_handlers.not_core_node_baselevel_handler
    terrainbento.boundary_handlers.single_node_baselevel_handler
    terrainbento.boundary_handlers.capture_node_baselevel_handler


Valid Landlab Components
------------------------

- `NormalFault <https://landlab.readthedocs.io/en/master/reference/components/normal_fault.html>`_.
