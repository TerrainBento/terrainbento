import os
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicHySa


def test_no_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001}

    pytest.raises(ValueError, BasicHySa, params=params)


def test_both_Ksp_or_Kss():
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': 200.,
              'regolith_transport_parameter': 0.001,
              'water_erodability': 0.001,
              'water_erodability~shear_stress': 0.001}
    pytest.raises(ValueError, BasicHySa, params=params)
    
def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    m = 0.5
    n = 1.0
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
    H_star = 0.1
    initial_soil_thickness = 0
    soil_transport_decay_depth = 1
    soil_production__maximum_rate = 0
    soil_production__decay_depth = 0.5
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
              'K_rock_sp': K_rock_sp,
              'K_sed_sp': K_sed_sp,
              'sp_crit_br': 0,
              'sp_crit_sed': 0,
              'm_sp': m,
              'n_sp': n,
              'v_sc': v_sc,
              'phi': phi,
              'F_f': F_f,
              'H_star': H_star,
              'solver': 'basic',
              'initial_soil_thickness': initial_soil_thickness,
              'soil_transport_decay_depth': soil_transport_decay_depth,
              'soil_production__maximum_rate': soil_production__maximum_rate,
              'soil_production__decay_depth': soil_production__decay_depth,
              'random_seed': 3141,
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = BasicHySa(params=params)
    for i in range(800):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes =  (np.power(((U * v_sc) / (K_sed_sp
        * np.power(actual_areas, m)))
        + (U / (K_rock_sp * np.power(actual_areas,
        m))), 1./n))

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]],
                              decimal=4)