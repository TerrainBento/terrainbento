"""
Base class models in the terrainbento package.

terrainbento includes to base classes useful for creating derived models. The
first of these base classes is ``_ErosionModel`` which is the base class for all
terrainbento models. ``_ErosionModel`` has extensive common functions useful for
model initialization, input, and output. The second base class,
``_StochasticErosionModel`` is a base class derived from ``_ErosionModel`` which
includes the common functions for models that use stochastic hydrology.
"""

from .erosion_model import ErosionModel
from .stochastic_erosion_model import StochasticErosionModel
