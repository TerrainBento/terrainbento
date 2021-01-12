"""Output writers

This text to be filled out...
"""

from .generic_output_writer import GenericOutputWriter
from .static_interval_writer import StaticIntervalOutputWriter
from .static_interval_adapters import StaticIntervalOutputClassAdapter
from .static_interval_adapters import StaticIntervalOutputFunctionAdapter

__all__ = [
    "GenericOutputWriter",
    "StaticIntervalOutputWriter",
    "StaticIntervalOutputClassAdapter",
    "StaticIntervalOutputFunctionAdapter",
]
