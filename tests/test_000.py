import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import Basic


def test_no_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001}

    pytest.raises(ValueError, Basic, params=params)


def test_both_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001,
              'water_erodability': 0.001,
              'water_erodability~shear_stress': 0.001}
    pytest.raises(ValueError, Basic, params=params)


def test_steady_Kss_no_precip_changer():
    U = 0.0001
    K = 0.001
    m = 1./3.
    n = 2./3.
    dt = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = Basic(params=params)
    for i in range(100):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]])


def test_steady_Ksp_no_precip_changer():
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
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = Basic(params=params)
    for i in range(100):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]])


def test_diffusion_only():
    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.75
    n = 1.0
    dt =1000

    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 21,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'west_boundary_closed': False,
              'south_boundary_closed': True,
              'regolith_transport_parameter': D,
              'water_erodability': 0.0,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}
    nts = int(total_time/dt)

    reference_node = 9
    # construct and run model
    model = Basic(params=params)
    for i in range(nts):
        model.run_one_step(dt)


    predicted_z = (model.z[model.grid.core_nodes[reference_node]]-(U / (2. * D)) *
               ((model.grid.x_of_node - model.grid.x_of_node[model.grid.core_nodes[reference_node]])**2))

    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(predicted_z[model.grid.core_nodes],
                              model.z[model.grid.core_nodes],
                              decimal=2)


def test_with_precip_changer():
    pass


def test_steady_m_075():
    U = 0.0001
    K = 0.001
    m = 0.75
    n = 1.0
    dt = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = Basic(params=params)
    for i in range(200):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]])

def test_steady_m_025():
    U = 0.0001
    K = 0.001
    m = 0.25
    n = 1.0
    dt = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = Basic(params=params)
    for i in range(200):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]])

def run_000_from_command_line_no_file():
    result = suprocess.call(['python' 'terrainbento/derived_models/model_000_basic.py'])
    assert result == 1

def run_000_from_command_line():
    subprocess.call('')
