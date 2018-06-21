import os
import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def output_writer_function_a(model):
    pass


def output_writer_function_b(model):
    pass


class output_writer_class_a(object):
    def __init__(model):
        self.model=model

    def run_one_step(self):
        pass


class output_writer_class_b(object):
    def __init__(model):
        self.model=model
    def run_one_step(self):
        pass


def output_writer_function_b(model):
    pass



def test_pickle_with_output_writers():
    pass


def test_two_function_writers():
    pass


def test_two_class_writers():
    pass


def test_all_four_writers():
    pass
