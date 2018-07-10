import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicSa

#test diffusion without stream power
def test_diffusion_only():
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    number_of_node_columns = 21
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    runtime = 100000

    #Construct dictionary. Note that stream power is turned off
    params = {'model_grid': 'RasterModelGrid',
                'dt': dt,
                'output_interval': 2.,
                'run_duration': 200.,
                'number_of_node_rows' : 3,
                'number_of_node_columns' : 21,
                'node_spacing' : dx,
                'north_boundary_closed': True,
                'south_boundary_closed': True,
                'regolith_transport_parameter': regolith_transport_parameter,
                'soil_transport_decay_depth': soil_transport_decay_depth,
                'max_soil_production_rate': max_soil_production_rate,
                'soil_production_decay_depth': soil_production_decay_depth,
                'initial_soil_thickness': initial_soil_thickness,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'depression_finder': 'DepressionFinderAndRouter',
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}


    #Construct and run model
    model = BasicSa(params=params)
    for i in range(runtime):
      model.run_one_step(dt)

    #test steady state soil depth
    actual_depth = model.grid.at_node['soil__depth'][30]
    predicted_depth = -soil_production_decay_depth*np.log(U/max_soil_production_rate)
    assert_array_almost_equal(actual_depth,predicted_depth,decimal = 3)

    #test steady state slope
    actual_profile = model.grid.at_node['topographic__elevation'][21:42]

    domain = np.arange(0, max(model.grid.node_x + dx), dx)
    steady_domain = np.arange(-max(domain)/2., max(domain)/2. + dx, dx)
    
    steady_z_profile_firsthalf = (steady_domain[0:len(domain)/2])**2*U/(regolith_transport_parameter*2*(1-np.exp(-predicted_depth/soil_transport_decay_depth)))-(U*(number_of_node_columns/2)**2)/(2*regolith_transport_parameter*(1-np.exp(-predicted_depth/soil_transport_decay_depth)))
    steady_z_profile_secondhalf = -(steady_domain[len(domain)/2:])**2*U/(regolith_transport_parameter*2*(1-np.exp(-predicted_depth/soil_transport_decay_depth)))+(U*(number_of_node_columns/2)**2)/(2*regolith_transport_parameter*(1-np.exp(-predicted_depth/soil_transport_decay_depth)))



    steady_z_profile = np.append([-steady_z_profile_firsthalf],[steady_z_profile_secondhalf])
    predicted_profile = steady_z_profile - np.min(steady_z_profile)

    assert_array_almost_equal(actual_profile,predicted_profile)

def test_field_in_grid
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    number_of_node_columns = 21
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    runtime = 100000
    params = {'model_grid': 'RasterModelGrid',
                'dt': dt,
                'output_interval': 2.,
                'run_duration': 200.,
                'number_of_node_rows' : 3,
                'number_of_node_columns' : 21,
                'node_spacing' : dx,
                'north_boundary_closed': True,
                'south_boundary_closed': True,
                'regolith_transport_parameter': regolith_transport_parameter,
                'soil_transport_decay_depth': soil_transport_decay_depth,
                'max_soil_production_rate': max_soil_production_rate,
                'soil_production_decay_depth': soil_production_decay_depth,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
                'depression_finder': 'DepressionFinderAndRouter',
                'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}


    model = BasicSa(params=params)

    model = BasicSa(params=params)





def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.001
    K = 0.01
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0.0
    soil_transport_decay_depth = 0.5
    run_time = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
                'dt': dt,
                'output_interval': 2.,
                'run_duration': 200.,
                'number_of_node_rows' : 3,
                'number_of_node_columns' : 20,
                'node_spacing' : dx,
                'north_boundary_closed': True,
                'south_boundary_closed': True,
                'regolith_transport_parameter': regolith_transport_parameter,
                'soil_transport_decay_depth': soil_transport_decay_depth,
                'max_soil_production_rate': max_soil_production_rate,
                'soil_production_decay_depth': soil_production_decay_depth,
                'initial_soil_thickness': 0.0,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'depression_finder': 'DepressionFinderAndRouter',
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}

    # construct and run model
    model = BasicSa(params=params)
    for i in range(run_time):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]], decimal = 4)



def test_steady_Ksp_no_precip_changer():
    U = 0.001
    K = 0.01
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    run_time = 1000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
                'dt': dt,
                'output_interval': 2.,
                'run_duration': 200.,
                'number_of_node_rows' : 3,
                'number_of_node_columns' : 20,
                'node_spacing' : dx,
                'north_boundary_closed': True,
                'south_boundary_closed': True,
                'regolith_transport_parameter': regolith_transport_parameter,
                'soil_transport_decay_depth': soil_transport_decay_depth,
                'max_soil_production_rate': max_soil_production_rate,
                'soil_production_decay_depth': soil_production_decay_depth,
                'initial_soil_thickness': 0.0,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}

    # construct and run model
    model = BasicSa(params=params)
    for i in range(run_time):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]], decimal = 4)


def test_with_precip_changer():
    K =  0.01
    U = 0.001
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    number_of_node_columns = 20
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
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
              'soil_transport_decay_depth': soil_transport_decay_depth,
              'max_soil_production_rate': max_soil_production_rate,
              'soil_production_decay_depth': soil_production_decay_depth,
              'initial_soil_thickness': 0.0,
              'slope_crit': 0.2,
              'water_erodability': K,
              'm_sp': 0.5,
              'n_sp': 1.0,
              'random_seed': 3141,
              'BoundaryHandlers': 'PrecipChanger',
              'PrecipChanger' : {'daily_rainfall__intermittency_factor': 0.5,
                                 'daily_rainfall__intermittency_factor_time_rate_of_change': 0.1,
                                 'daily_rainfall__mean_intensity': 1.0,
                                 'daily_rainfall__mean_intensity_time_rate_of_change': 0.2}}

    model = BasicSa(params=params)
    assert model.eroder.K == K
    assert 'PrecipChanger' in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K, 5) == 0.10326



