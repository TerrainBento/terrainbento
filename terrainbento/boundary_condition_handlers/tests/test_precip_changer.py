from numpy.testing import assert_almost_equal
from nose.tools import assert_raises

from terrainbento.boundary_condition_handlers import _depth_to_intensity

def test_bad_time_unit():
    "Test a bad time unit in PrecipChanger"
    assert_raises(NotImplementedError, _depth_to_intensity, 2.0, 'foo')
