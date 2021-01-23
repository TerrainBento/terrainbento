"""Output writers

This text to be filled out...
"""

from .generic_output_writer import (
    GenericOutputWriter,
    OutputIteratorSkipWarning,
)
from .static_interval_writer import StaticIntervalOutputWriter
from .static_interval_adapters import (
    StaticIntervalOutputClassAdapter,
    StaticIntervalOutputFunctionAdapter,
)
from .ow_simple_netcdf import OWSimpleNetCDF


__all__ = [
    "GenericOutputWriter",
    "StaticIntervalOutputWriter",
    "StaticIntervalOutputClassAdapter",
    "StaticIntervalOutputFunctionAdapter",
    "OutputIteratorSkipWarning",
    "OWSimpleNetCDF",
]
