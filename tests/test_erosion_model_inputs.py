import os
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from landlab import HexModelGrid
from landlab.components import FlowAccumulator, DepressionFinderAndRouter
from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

at_node_fields = ['topographic__elevation',
                  'initial_topographic__elevation',
                  'cumulative_erosion__depth',
                  'water__unit_flux_in',
                  'flow__receiver_node',
                  'topographic__steepest_slope',
                  'flow__link_to_receiver_node',
                  'flow__sink_flag', 'drainage_area',
                  'surface_water__discharge',
                  'flow__upstream_node_order',
                  'flow__data_structure_delta']


def test_no_inputs():
    pytest.raises(ValueError, ErosionModel)


def test_both_inputs():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    fp = os.path.join(_TEST_DATA_DIR, 'inputs.txt')

    pytest.raises(ValueError, ErosionModel, params=params, input_file=fp)

def test_both_node_rows_and_DEM():
    params = {'model_grid': 'HexModelGrid',
              'dt': 1,
              'output_interval': 2.,
              'number_of_node_rows': 5,
              'DEM_filename': 'foo.nc'}
    pytest.raises(ValueError, ErosionModel, params=params)

def test_no_required_params():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2.}
    pytest.raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'run_duration': 10.}
    pytest.raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'output_interval': 2, 'run_duration': 10.}
    pytest.raises(ValueError, ErosionModel, params=params)


def test_bad_req_params():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 'spam', 'output_interval': 2., 'run_duration': 10.}
    pytest.raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 'eggs', 'run_duration': 10.}
    pytest.raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 'wooo'}
    pytest.raises(ValueError, ErosionModel, params=params)


def test_input_file():
    fp = os.path.join(_TEST_DATA_DIR, 'inputs.txt')
    em = ErosionModel(input_file=fp)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert em.flow_director == 'FlowDirectorSteepest'
    assert isinstance(em.flow_accumulator, FlowAccumulator) == True
    assert em.depression_finder is None
    assert em.boundary_handler == {}
    assert em.output_writers == {'class': {}, 'function': []}
    assert em.save_first_timestep == True
    assert em._out_file_name == 'terrainbento_output'
    assert em._model_time == 0.


def test_parameters():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert em.flow_director == 'FlowDirectorSteepest'
    assert isinstance(em.flow_accumulator, FlowAccumulator) == True
    assert em.depression_finder is None
    assert em.boundary_handler == {}
    assert em.output_writers == {'class': {}, 'function': []}
    assert em.save_first_timestep == True
    assert em._out_file_name == 'terrainbento_output'
    assert em._model_time == 0.
