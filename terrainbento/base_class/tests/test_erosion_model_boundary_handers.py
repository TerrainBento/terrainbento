import os
import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_bad_output_writer_instance():
    pass


def test_bad_output_writer_string():
    pass


def test_boundary_condition_handler_with_special_part_of_params():
    pass


def test_pickle_with_boundary_handlers():
    pass


def test_example_boundary_handlers():
    pass


def test_pass_boundary_handlers_as_str():
    pass


def test_pass_boundary_handlers_as_instance():
    pass


def test_pass_two_boundary_handlers():
    pass
