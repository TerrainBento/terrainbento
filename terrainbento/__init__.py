"""Public classes of the terrainbento package.

terrainbento has five types of public classes: base models, derived models
precipitators, runoff-generators, and boundary condition handlers. Base models
are used to help easily build new models. Derived models are models that have
inherited from the **ErosionModel** base class. Precipitators can make different
types of spatially variable precipitation. Runoff-generators convert
precipiation to runoff and Boundary condition handlers are helper classes that
have been designed to modify model boundary conditions during a model run.
"""

from ._version import get_versions
from .base_class import (
    ErosionModel,
    StochasticErosionModel,
    TwoLithologyErosionModel,
)
from .boundary_handlers import (
    CaptureNodeBaselevelHandler,
    GenericFuncBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
)
from .clock import Clock
from .derived_models import (
    Basic,
    BasicCh,
    BasicChRt,
    BasicChRtTh,
    BasicChSa,
    BasicCv,
    BasicDd,
    BasicDdHy,
    BasicDdRt,
    BasicDdSt,
    BasicDdVs,
    BasicHy,
    BasicHyRt,
    BasicHySa,
    BasicHySt,
    BasicHyVs,
    BasicRt,
    BasicRtSa,
    BasicRtTh,
    BasicRtVs,
    BasicSa,
    BasicSaVs,
    BasicSt,
    BasicStTh,
    BasicStVs,
    BasicTh,
    BasicThVs,
    BasicVs,
)
from .model_template import ModelTemplate
from .precipitators import RandomPrecipitator, UniformPrecipitator
from .runoff_generators import SimpleRunoff

__all__ = [
    "Clock",
    "ModelTemplate",
    "Basic",
    "BasicTh",
    "BasicDd",
    "BasicHy",
    "BasicCh",
    "BasicSt",
    "BasicVs",
    "BasicSa",
    "BasicRt",
    "BasicCv",
    "BasicDdHy",
    "BasicStTh",
    "BasicDdSt",
    "BasicHySt",
    "BasicThVs",
    "BasicDdVs",
    "BasicStVs",
    "BasicHySa",
    "BasicHyVs",
    "BasicChSa",
    "BasicSaVs",
    "BasicRtTh",
    "BasicDdRt",
    "BasicHyRt",
    "BasicChRt",
    "BasicRtVs",
    "BasicRtSa",
    "BasicChRtTh",
    "UniformPrecipitator",
    "RandomPrecipitator",
    "SimpleRunoff",
    "CaptureNodeBaselevelHandler",
    "NotCoreNodeBaselevelHandler",
    "SingleNodeBaselevelHandler",
    "GenericFuncBaselevelHandler",
    "PrecipChanger",
    "ErosionModel",
    "StochasticErosionModel",
    "TwoLithologyErosionModel",
]


__version__ = get_versions()["version"]
del get_versions
