import numpy as np
#from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises#, assert_almost_equal, assert_equal

from landlab import HexModelGrid
from terrainbento import ErosionModel


def test_check_walltime():
    pass

def test_length_conversion_raises_error():
    # test that passing both results in an error
    params = {'model_grid' : 'HexModelGrid',
              'meters_to_feet' : True,
              'feet_to_meters' : True,
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    assert_raises(ValueError,
                  ErosionModel,
                  params=params)

def test_meters_to_feet_correct():
    # first meters_to_feet
    params = {'model_grid' : 'HexModelGrid',
              'meters_to_feet' : True,
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert em._length_factor == 3.28084


def test_feet_to_meters_correct():
    params = {'model_grid' : 'HexModelGrid',
              'feet_to_meters' : True,
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert em._length_factor == 1.0/3.28084


def test_no_units_correct():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert em._length_factor == 1.0


def test_calc_cumulative_erosion():
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert np.array_equiv(em.z, 0.) == True
    em.z += 1.
    em.calculate_cumulative_change()
    assert np.array_equiv(em.grid.at_node['cumulative_erosion__depth'], 1.) == True


def test_parameter_exponent_both_provided():
    """Test the get_parameter_from_exponent function when both are provided."""
    params = {'model_grid' : 'HexModelGrid',
              'water_erodability_exp' : -3.,
              'water_erodability' : 0.01,
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert_raises(ValueError,
                  em.get_parameter_from_exponent,
                  'water_erodability')


def test_parameter_exponent_neither_provided():
    """Test the get_parameter_from_exponent function when neither are provided."""
    params = {'model_grid' : 'HexModelGrid',
              'dt': 1, 'output_interval': 2., 'run_duration': 10.}
    em = ErosionModel(params=params)
    assert_raises(ValueError,
                  em.get_parameter_from_exponent,
                  'water_erodability')
    val = em.get_parameter_from_exponent('water_erodability', raise_error=False)
    assert val is None
