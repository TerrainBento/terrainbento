#! /usr/bin/env python
"""The complete Basic Model Interface."""

from .bmi import Bmi
from .base import BmiBase
from .info import BmiInfo
from .time import BmiTime
from .vars import BmiVars
from .getter_setter import BmiGetter, BmiSetter
from .grid_rectilinear import BmiGridRectilinear
from .grid_uniform_rectilinear import BmiGridUniformRectilinear
from .grid_structured_quad import BmiGridStructuredQuad
from .grid_unstructured import BmiGridUnstructured


class BmiModel(Bmi, BmiBase, BmiInfo, BmiTime, BmiVars, BmiGetter, BmiSetter,
          BmiGridRectilinear, BmiGridUniformRectilinear, BmiGridStructuredQuad,
          BmiGridUnstructured):

    """The complete Basic Model Interface for terrainbento.

    Defines an interface for converting a standalone model into an
    integrated modeling framework component.
    """

    def __init__(self, clock, grid):
        # save the grid, clock, and parameters.
        self._grid = grid
        self.clock = clock
