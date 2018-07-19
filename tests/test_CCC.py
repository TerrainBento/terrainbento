import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,

from landlab import HexModelGrid
from terrainbento import BasicCv



def test_steady_Ksp_with_depression_finding():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    dt = 10
    climate_factor = 0.5
    climate_constant_date = 10
    run_time = 100
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': run_time*dt,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'climate_factor': climate_factor,
              'climate_constant_date': climate_constant_date,
              'water_erodability': K,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'depression_finder': 'DepressionFinderAndRouter',
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}

    # construct and run model
    model = BasicCv(params=params)
    for i in range(run_time):
        model.run_one_step(dt)




def test_diffusion_only():
    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.75
    n = 1.0
    dt =1000
    climate_factor = 0.5
    climate_constant_date = 10

    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'run_duration': total_time,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 21,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'west_boundary_closed': False,
              'south_boundary_closed': True,
              'regolith_transport_parameter': D,
              'climate_factor': climate_factor,
              'climate_constant_date': climate_constant_date,
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
    model = BasicCv(params=params)
    for i in range(nts):
        model.run_one_step(dt)


    predicted_z = (model.z[model.grid.core_nodes[reference_node]]-(U / (2. * D)) *
               ((model.grid.x_of_node - model.grid.x_of_node[model.grid.core_nodes[reference_node]])**2))


    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(predicted_z[model.grid.core_nodes],
                              model.z[model.grid.core_nodes],
                              decimal=2)


def test_with_precip_changer():
    K =  0.01
    climate_factor = 0.5
    climate_constant_date = 10
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
              'climate_factor': climate_factor,
              'climate_constant_date': climate_constant_date,
              'water_erodability': K,
              'm_sp': 0.5,
              'n_sp': 1.0,
              'random_seed': 3141,
              'BoundaryHandlers': 'PrecipChanger',
              'PrecipChanger' : {'daily_rainfall__intermittency_factor': 0.5,
                                 'daily_rainfall__intermittency_factor_time_rate_of_change': 0.1,
                                 'daily_rainfall__mean_intensity': 1.0,
                                 'daily_rainfall__mean_intensity_time_rate_of_change': 0.2}}

    model = BasicCv(params=params)
    #assert model.eroder.K == K
    assert 'PrecipChanger' in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    #assert round(model.eroder.K, 5) == 0.10326
