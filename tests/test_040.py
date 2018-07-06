import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicCh


def test_diffusion_only():
    U = 0.0001
    K = 0.0
    m = 0.5
    n = 1.0
    dt = 10
    D = 0.001
    S_c = 0.2
    dx = 10.0

    #Construct dictionary. Note that stream power is turned off
    params = {'model_grid': 'RasterModelGrid',
                'dt': dt,
                'output_interval': 2.,
                'run_duration': 200.,
                'number_of_node_rows' : 3,
                'number_of_node_columns' : 21,
                'node_spacing' : dx,
                'regolith_transport_parameter': D,
                'water_erodability': K,
                'm_sp': m,
                'n_sp': n,
          	  'slope_crit': S_c,
          	  'depression_finder': 'DepressionFinderAndRouter',
          	  'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
                'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                                'lowering_rate': -U}}

    # #Construct and run model
    # model = BasicCh(params=params)
    # for i in range(100):
    #   model.run_one_step(dt)



    #Construct actual and predicted slope at right edge of domain
    x = 9*dx
    qs = U*dx
    nterms = 11
    for i in range(1,nterms+1):
      p[2*i-2] = D*(1/(S_c**(2*(i-1))))
    p = np.fliplr([p])[0]
    p = np.append(p,qs)
    p_roots = np.roots(p)
    predicted_slope = np.real(p_roots[-1])

    actual_slope = model.grid.at_node['topographic__steepest_slope'][39]
    assert_array_almost_equal(actual_slope, predicted_slope)

def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    dt = 1000
    run_time = 10000
    # construct dictionary. note that D is turned off here
    params = {'model_grid': 'RasterModelGrid',
              'dt': dt,
              'output_interval': 2.,
              'run_duration': 200.,
              'number_of_node_rows' : 3,
              'number_of_node_columns' : 20,
              'node_spacing' : 100.0,
              'north_boundary_closed': True,
              'south_boundary_closed': True,
              'regolith_transport_parameter': 0.,
              'water_erodability': K,
              'slope_crit': 0.0,
              'm_sp': m,
              'n_sp': n,
              'random_seed': 3141,
              'depression_finder': 'DepressionFinderAndRouter',
              'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
              'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
                                              'lowering_rate': -U}}
    
    # construct and run model
    model = BasicCh(params=params)
    for i in range(run_time):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node['topographic__steepest_slope']
    actual_areas = model.grid.at_node['drainage_area']
    predicted_slopes = (U/(K * (actual_areas**m))) ** (1./n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes[model.grid.core_nodes[1:-1]],
                              predicted_slopes[model.grid.core_nodes[1:-1]])





















