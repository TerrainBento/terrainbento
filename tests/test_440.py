import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicChSa

#test diffusion without stream power
def test_diffusion_only():
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    dt = 1
    dx = 10.0
    number_of_node_columns = 21
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    S_c = 0.2
    runtime = 50000

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
                'soil_production__maximum_rate': max_soil_production_rate,
                'soil_production__decay_depth': soil_production_decay_depth,
                'soil__initial_thickness': initial_soil_thickness,
                'critical_slope': S_c,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'depression_finder': 'DepressionFinderAndRouter',
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}


    #Construct and run model
    model = BasicChSa(params=params)
    for i in range(runtime):
      model.run_one_step(dt)

    #test steady state soil depth
    actual_depth = model.grid.at_node['soil__depth'][30]
    predicted_depth = -soil_production_decay_depth*np.log(U/max_soil_production_rate)
    assert_array_almost_equal(actual_depth,predicted_depth,decimal = 3)

    #Construct actual and predicted slope at right edge of domain
    x = 8.5*dx
    qs = U*x
    nterms = 11
    p = np.zeros(2*nterms-1)
    for k in range(1,nterms+1):
      p[2*k-2] = regolith_transport_parameter*(1-np.exp(-predicted_depth/soil_transport_decay_depth))*(1/(S_c**(2*(k-1))))
    p = np.fliplr([p])[0]
    p = np.append(p,qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    #print(predicted_slope)

    actual_slope = np.abs(model.grid.at_node['topographic__steepest_slope'][39])
    #print model.grid.at_node['topographic__steepest_slope']
    assert_array_almost_equal(actual_slope, predicted_slope, decimal = 3)


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.001
    K = 0.01
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0
    S_c = 0.2
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
                'soil_production__maximum_rate': max_soil_production_rate,
                'soil_production__decay_depth': soil_production_decay_depth,
                'soil__initial_thickness': 0.0,
                'critical_slope': S_c,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'depression_finder': 'DepressionFinderAndRouter',
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}

    # construct and run model
    model = BasicChSa(params=params)
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
    S_c = 0.2
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
                'soil_production__maximum_rate': max_soil_production_rate,
                'soil_production__decay_depth': soil_production_decay_depth,
                'soil__initial_thickness': 0.0,
                'critical_slope': S_c,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	    'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}

    # construct and run model
    model = BasicChSa(params=params)
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
    S_c = 0.2
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
              'soil_production__maximum_rate': max_soil_production_rate,
              'soil_production__decay_depth': soil_production_decay_depth,
              'soil__initial_thickness': 0.0,
              'critical_slope': S_c,
              'critical_slope': 0.2,
              'water_erodability': K,
              'm_sp': 0.5,
              'n_sp': 1.0,
              'random_seed': 3141,
              'BoundaryHandlers': 'PrecipChanger',
              'PrecipChanger' : {'daily_rainfall__intermittency_factor': 0.5,
                                 'daily_rainfall__intermittency_factor_time_rate_of_change': 0.1,
                                 'daily_rainfall__mean_intensity': 1.0,
                                 'daily_rainfall__mean_intensity_time_rate_of_change': 0.2}}

    model = BasicChSa(params=params)
    assert model.eroder.K == K
    assert 'PrecipChanger' in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K, 5) == 0.10326
