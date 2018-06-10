import os
import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import Basic

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_run_for():
    fp = os.path.join(_TEST_DATA_DIR, 'basic_inputs.txt')
    model = Basic(input_file=fp)
    model.run_for(10., 100.)
    assert model.model_time == 100.

def test_finalize():
    pass

def test_run():
    pass
