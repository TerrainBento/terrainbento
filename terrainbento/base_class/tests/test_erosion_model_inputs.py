import os
import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_no_inputs():
    """Test providing neither input file nor parameters."""
    pass

def test_both_inputs():
    """Test providing both input file and parameters."""
    pass

def test_no_required_params():
    pass

def test_bad_req_params():
    pass

def test_input_file():
    """Test providing input file."""
    pass

def test_parameters():
    """Test providing parameters file."""
    pass

def test_load_from_pickle():
    """Test providing parameters file."""
    pass

def test_create_pickle():
    """Test providing parameters file."""
    pass
