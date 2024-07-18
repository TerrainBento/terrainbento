"""Output writers

This text to be filled out...
"""

from .generic_output_writer import GenericOutputWriter, OutputIteratorSkipWarning
from .ow_simple_netcdf import OWSimpleNetCDF
from .static_interval_adapters import (
    StaticIntervalOutputClassAdapter,
    StaticIntervalOutputFunctionAdapter,
)
from .static_interval_writer import StaticIntervalOutputWriter

__all__ = [
    "GenericOutputWriter",
    "StaticIntervalOutputWriter",
    "StaticIntervalOutputClassAdapter",
    "StaticIntervalOutputFunctionAdapter",
    "OutputIteratorSkipWarning",
    "OWSimpleNetCDF",
]
