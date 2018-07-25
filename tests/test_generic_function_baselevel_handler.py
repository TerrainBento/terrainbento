# coding: utf8
#! /usr/env/python
import numpy as np

import pytest

from landlab import RasterModelGrid, HexModelGrid

from terrainbento.boundary_condition_handlers import GenericFuncBaselevelHandler


def test_function_of_four_variables():
    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        GenericFuncBaselevelHandler(mg, function=lambda x, y, t, q: (10*x + 10*y + 10*t, + 10*q))

def test_function_that_returns_wrong_size():
    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        GenericFuncBaselevelHandler(mg, function=lambda x, y, t: np.mean(10*x + 10*y + 10*t))
