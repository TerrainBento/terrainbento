import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel


def test_FlowAccumulator_options():
    """Test FlowAccumulator instantiation."""
    # D8, D4, MFD x hex, raster, dem
    # with/without DepressionFinder
    pass
