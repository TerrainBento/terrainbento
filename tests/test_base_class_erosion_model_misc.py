# coding: utf8
# !/usr/env/python

import numpy as np

# from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from terrainbento import ErosionModel
from terrainbento.utilities import filecmp


def test_length_conversion_raises_error(clock_simple):
    # test that passing both results in an error
    params = {
        "model_grid": "HexModelGrid",
        "meters_to_feet": True,
        "feet_to_meters": True,
        "clock": clock_simple,
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)


def test_meters_to_feet_correct(clock_simple):
    # first meters_to_feet
    params = {
        "model_grid": "HexModelGrid",
        "meters_to_feet": True,
        "clock": clock_simple,
    }
    em = ErosionModel(params=params)
    assert em._length_factor == 3.28084


def test_feet_to_meters_correct(clock_simple):
    params = {
        "model_grid": "HexModelGrid",
        "feet_to_meters": True,
        "clock": clock_simple,
    }
    em = ErosionModel(params=params)
    assert em._length_factor == 1.0 / 3.28084


def test_no_units_correct(clock_simple):
    params = {"model_grid": "HexModelGrid", "clock": clock_simple}
    em = ErosionModel(params=params)
    assert em._length_factor == 1.0


def test_calc_cumulative_erosion(clock_simple):
    params = {"model_grid": "HexModelGrid", "clock": clock_simple}
    em = ErosionModel(params=params)
    assert np.array_equiv(em.z, 0.) is True
    em.z += 1.
    em.calculate_cumulative_change()
    assert (
        np.array_equiv(em.grid.at_node["cumulative_elevation_change"], 1.)
        is True
    )
