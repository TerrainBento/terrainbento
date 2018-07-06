import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicCh

def test_no_slope_crit():
	params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        pytest.raises(ValueError,BasicCh,params=params)


def test_steady_diffusion():
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
        ...           'dt': dt,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : dx,
        ...           'regolith_transport_parameter': D,
        ...           'water_erodability': K,
        ...           'm_sp': m,
        ...           'n_sp': n,
        			  'slope_crit': S_c,
        			  'depression_finder': 'DepressionFinderAndRouter',
        			  'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
        			  'NotCoreNodeBaselevelHandler': {'modify_core_node': True,
        			  								  'lowering_rate': -U}}

    #Construct and run model
    model = BasicCh(params=params)
    for i in range(100):
    	model.run_one_step(dt)

    #Construct actual and predicted slope at right edge of domain
    x = 3*dx
    qs = U*dx
    nterms = 11
    for i in range(1,nterms+1):
      p[2*i-2] = D*(1/(S_c**(2*(i-1))))
    p = np.fliplr([p])[0]
    p = np.append(p,qs)
    p_roots = np.roots(p)
    predicted_slope = np.real(p_roots[-1])

    actual_slope = model.grid.at_node['topographic__steepest_slope'][17]
    assert_array_almost_equal(actual_slope, predicted_slope)





















