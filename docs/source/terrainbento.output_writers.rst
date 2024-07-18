.. py:module:: terrainbento.output_writers

Output Writers
==============

The terrainbento output writers permit advanced control over writing a variety
of output types at user specified time intervals. This module was greatly
enhanced by `@alexmitchell`_.

.. _@alexmitchell: https://github.com/alexmitchell

Note: The output writer classes require that the output time points specified
line up with the model timesteps as specified by the Clock. If the timepoints
do not line up, warnings are raised the first four times an output time is
skipped. On the fifth time an error is raised.

.. toctree::

    terrainbento.output_writers.generic_output_writer
    terrainbento.output_writers.static_interval_writer
    terrainbento.output_writers.ow_simple_netcdf
    terrainbento.output_writers.static_interval_adapters
