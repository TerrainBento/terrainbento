import sys
import os

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal
from nose.tools import assert_raises

from terrainbento.boundary_condition_handlers import ClosedNodeBaselevelHandler
from landlab import RasterModelGrid, HexModelGrid

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def text_hex():
    "Test using a hex grid"

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')
    
    bh = ClosedNodeBaselevelHandler(mg, modify_closed_nodes = False, lowering_rate = -0.1)
    bh.run_one_step(10.0)
    
    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0 
    
    # closed should have stayed the same
    assert_array_equal(z[closed], np.zeros(np.sum(closed)))
    
    # not closed should have been uplifted 10*0.1
    assert_array_equal(z[not_closed], np.ones(np.sum(not_closed)))


def test_passing_neither_lowering_method():
    """Test passing no lowering information"""
    mg = RasterModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')

    assert_raises(ValueError,
                  ClosedNodeBaselevelHandler,
                  mg, modify_closed_nodes = True)


def test_passing_both_lowering_methods():
    """Test passing both lowering methods"""
    mg = RasterModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')
    file = os.path.join(_TEST_DATA_DIR, 'outlet_history.txt')

    assert_raises(ValueError,
                  ClosedNodeBaselevelHandler,
                  mg, modify_closed_nodes = False,
                  lowering_rate = -0.1,
                  lowering_file_path=file)


def test_outlet_lowering_object_bad_file():
    """Test using an outlet lowering object with a bad file"""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')

    assert_raises(ValueError,
                  ClosedNodeBaselevelHandler,
                  mg,
                  modify_closed_nodes = False,
                  lowering_file_path='foo.txt')


def test_outlet_lowering_rate_no_scaling_bedrock():
    """Test using an outlet lowering rate with no scaling and bedrock"""

    mg = RasterModelGrid(5, 5)
    z = mg.add_ones('node', 'topographic__elevation')
    b = mg.add_zeros('node', 'bedrock__elevation')

    bh = ClosedNodeBaselevelHandler(mg,
                                    modify_closed_nodes = False,
                                    lowering_rate = -0.1)
    bh.run_one_step(2400.0)
    
    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0 
    
    # closed should have stayed the same
    assert_array_equal(z[closed], np.ones(np.sum(closed)))
    assert_array_equal(b[closed], np.zeros(np.sum(closed)))

    # not closed should have been uplifted 2400*0.1
    assert_array_equal(b[not_closed], 240.0 * np.ones(np.sum(not_closed)))
    assert_array_equal(z[not_closed], 241.0 * np.ones(np.sum(not_closed)))


def test_outlet_lowering_object_no_scaling_bedrock():
    """Test using an outlet lowering object with no scaling and bedrock"""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones('node', 'topographic__elevation')
    b = mg.add_zeros('node', 'bedrock__elevation')


    file = os.path.join(_TEST_DATA_DIR, 'outlet_history.txt')
    bh = ClosedNodeBaselevelHandler(mg,
                                    modify_closed_nodes = True,
                                    lowering_file_path=file)
    bh.run_one_step(2400.0)
    
    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0 
    
    # closed should have lowered -46.5
    assert_array_equal(z[closed], -46.5 * np.ones(np.sum(closed)))
    assert_array_equal(b[closed], -47.5 * np.ones(np.sum(closed)))

    # not closed should stayed the same
    assert_array_equal(z[not_closed], np.ones(np.sum(not_closed)))
    assert_array_equal(b[not_closed], np.zeros(np.sum(not_closed)))

def test_outlet_lowering_object_no_scaling():
    """Test using an outlet lowering object with no scaling"""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones('node', 'topographic__elevation')
    file = os.path.join(_TEST_DATA_DIR, 'outlet_history.txt')
    bh = ClosedNodeBaselevelHandler(mg,
                                    modify_closed_nodes = False,
                                    lowering_file_path=file)
    bh.run_one_step(2400.0)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0 
    
    # closed should have stayed the same
    assert_array_equal(z[closed], np.ones(np.sum(closed)))

    # not closed should raised by 47.5  to 48.5
    assert_array_equal(z[not_closed], 48.5 * np.ones(np.sum(not_closed)))


def test_outlet_lowering_object_with_scaling():
    """Test using an outlet lowering object with scaling"""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')
    file = os.path.join(_TEST_DATA_DIR, 'outlet_history.txt')
    bh = ClosedNodeBaselevelHandler(mg,
                                    modify_closed_nodes = True,
                                    lowering_file_path=file,
                                    model_end_elevation = -318.0)
    bh.run_one_step(2400.0)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0 
    
    # closed should have lowered -46.5
    assert_array_equal(z[closed], -95.0 * np.ones(np.sum(closed)))

    # not closed should stayed the same
    assert_array_equal(z[not_closed], np.zeros(np.sum(not_closed)))