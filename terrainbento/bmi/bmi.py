#! /usr/bin/env python
"""The complete Basic Model Interface."""

from bmipy import Bmi
from .base import BmiBase
from .info import BmiInfo
from .time import BmiTime
from .vars import BmiVars
from .getter_setter import BmiGetter, BmiSetter
from .grid import BmiGrid


class BmiModel(BmiBase, BmiInfo, BmiTime, BmiVars, BmiGetter, BmiSetter,
          BmiGrid, Bmi):

    """The complete Basic Model Interface for terrainbento.

    Defines an interface for converting a standalone model into an
    integrated modeling framework component.
    """

    def __init__(self, clock, grid):
        # save the grid, clock, and parameters.

        self._grid = grid
        self.clock = clock
