# coding: utf8
# !/usr/env/python

import numpy as np

from terrainbento import ErosionModel


def test_calc_cumulative_erosion(clock_simple, grid_1):
    em = ErosionModel(grid=grid_1, clock=clock_simple)
    assert np.array_equiv(em.z, 0.0) is True
    em.z += 1.0
    em.calculate_cumulative_change()
    assert (
        np.array_equiv(em.grid.at_node["cumulative_elevation_change"], 1.0)
        is True
    )
