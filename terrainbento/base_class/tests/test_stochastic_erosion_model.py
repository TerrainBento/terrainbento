import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import StochasticErosionModel


with/without Precip Changer

def test_reset_random_seed():
    pass


def test_finalize_opt_duration_stochastic_true():
    pass


def test_finalize_opt_duration_stochastic_false():
    pass


# double check if these two options work with BOTH stochastic duration options.
def test_write_storm_sequence_to_file():
    # this works with both
    pass


def test_write_exceedance_frequency_file():
    # this with stochastic duration = False.
    pass
