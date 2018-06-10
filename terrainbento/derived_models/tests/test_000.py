import os
import numpy as np
import glob

#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import Basic


def test_no_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001}

    assert_raises(ValueError, Basic, params=params)


def test_both_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001,
              'water_erodability': 0.001,
              'water_erodability~shear_stress': 0.001}
    assert_raises(ValueError, Basic, params=params)


def test_steady_Kss_no_precip_changer():
    pass


def test_steady_Ksp_no_precip_changer():
    pass


def test_diffusion_only():
    pass


def test_with_precip_changer():
    pass


def test_steady_var_m():
    pass
