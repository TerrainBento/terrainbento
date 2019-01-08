# coding: utf8
# !/usr/env/python

import numpy as np
import pytest

from terrainbento import ErosionModel


def test_calc_cumulative_erosion(clock_simple):
    params = {"model_grid": "HexModelGrid", "clock": clock_simple}
    em = ErosionModel(params=params)
    assert np.array_equiv(em.z, 0.) is True
    em.z += 1.
    em.calculate_cumulative_change()
    assert np.array_equiv(em.grid.at_node["cumulative_elevation_change"], 1.) is True
