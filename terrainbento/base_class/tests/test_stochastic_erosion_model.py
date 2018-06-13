import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import StochasticErosionModel, BasicSt


def test_run_opt_true():
    pass

def test_run_opt_false():
    pass

def test_run_opt_true_with_changer():
    pass

def test_run_opt_false_with_changer():
    pass

def test_reset_random_seed():
    pass


def test_finalize_opt_duration_stochastic_true():
    pass


def test_finalize_opt_duration_stochastic_false():
    pass

def test_float_number_of_sub_time_steps():
    pass

# double check if these two options work with BOTH stochastic duration options.
def test_write_storm_sequence_to_file():
    # this works with both
    pass


def test_write_exceedance_frequency_file():
    # this with stochastic duration = False.
    pass

def test_not_specifying_record_rain():
    pass

def test_opt_true_freq_file():
    pass
    # this raises a value error


def test_write_files_no_record():
    pass
    # both of these raise value errors.
