#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:58:16 2018

@author: barnhark
"""

import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel

def test_length_conversion():
    """Test meters to feet, and reverse."""
    # and that passing both gives an error
    pass

def test_calc_cumulative_erosion():
    pass

def test_parameter_exponent_both_provided():
    """Test the get_parameter_from_exponent function when both are provided."""
    params = {'model_grid' : 'HexModelGrid',
              'water_erodability_exp' : -3.,
              'water_erodability' : 0.01,
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert_raises(ValueError,
                  em.get_parameter_from_exponent,
                  'water_erodability')

def test_parameter_exponent_neither_provided():
    """Test the get_parameter_from_exponent function when neither are provided."""
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert_raises(ValueError,
                  em.get_parameter_from_exponent,
                  'water_erodability')

    val = em.get_parameter_from_exponent('water_erodability', raise_error=False)
    assert val is None
