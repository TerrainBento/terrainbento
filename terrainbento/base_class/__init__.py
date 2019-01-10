"""Base class models in the terrainbento package.

terrainbento includes two base classes useful for creating derived models. The
first of these base classes is **ErosionModel** which is the base class for all
terrainbento models. **ErosionModel** has extensive common functions useful for
model initialization, input, and output. The second base class,
**StochasticErosionModel** is a base class derived from **ErosionModel** which
includes the common functions for models that use stochastic hydrology.
**TwoLithologyErosionModel** handles reading in a contact zone elevation used
by all two-lithology models.
"""

from .erosion_model import ErosionModel
from .stochastic_erosion_model import StochasticErosionModel
from .two_lithology_erosion_model import TwoLithologyErosionModel

__all__ = [
    "ErosionModel",
    "StochasticErosionModel",
    "TwoLithologyErosionModel",
]
