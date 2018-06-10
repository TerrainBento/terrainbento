import os
import numpy as np
import glob

from numpy.testing import assert_array_almost_equal # assert_array_equal,
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
    # define parameters
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    dt = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 6,
              'number_of_node_columns' : 9,
              'node_spacing' : 10.0,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141}

    # construct and run model
    model = Basic(params=params)
    for i in range(100):
        model.run_one_step(dt)
        model.z[model.grid.core_nodes] += U*dt

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes],
                              predicted_slopes[model.grid.core_nodes])


def test_diffusion_only():
    pass


def test_with_precip_changer():
    pass


def test_steady_var_m():
    pass
