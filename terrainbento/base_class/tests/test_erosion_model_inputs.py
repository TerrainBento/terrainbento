import os
import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_no_inputs():
    assert_raises(ValueError, ErosionModel)


def test_both_inputs():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    fp = os.path.join(_TEST_DATA_DIR, 'inputs.txt')

    assert_raises(ValueError, ErosionModel, params=params, input_file=fp)


def test_no_required_params():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2.}
    assert_raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'run_duration': 10.}
    assert_raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'output_interval': 2, 'run_duration': 10.}
    assert_raises(ValueError, ErosionModel, params=params)


def test_bad_req_params():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 'spam', 'output_interval': 2., 'run_duration': 10.}
    assert_raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 'eggs', 'run_duration': 10.}
    assert_raises(ValueError, ErosionModel, params=params)

    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 'wooo'}
    assert_raises(ValueError, ErosionModel, params=params)


def test_input_file():
    fp = os.path.join(_TEST_DATA_DIR, 'inputs.txt')
    em = ErosionModel(input_file=fp)


def test_parameters():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)


def test_load_from_pickle():
    pass

def test_create_pickle():
    pass
