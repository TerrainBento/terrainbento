# coding: utf8
#! /usr/env/python

from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

from terrainbento.boundary_condition_handlers import PrecipChanger
from landlab import RasterModelGrid, HexModelGrid


def test_not_passing_daily_rainfall__intermittency_factor():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001 ,
            daily_rainfall__mean_intensity=3.0 ,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2 ,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=2.0 ,
        )


def test_not_passing_daily_rainfall__intermittency_factor_time_rate_of_change():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__mean_intensity=3.0 ,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2 ,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=2.0 ,
        )


def test_not_passing_daily_rainfall__mean_intensity():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001 ,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2 ,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=2.0 ,
        )


def test_not_passing_daily_rainfall__mean_intensity_time_rate_of_change():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001 ,
            daily_rainfall__mean_intensity=3.0 ,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=2.0 ,
        )


def test_not_passing_daily_rainfall__precipitation_shape_factor():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001 ,
            daily_rainfall__mean_intensity=3.0 ,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2 ,
            infiltration_capacity=2.0 ,
        )


def test_not_passing_infiltration_capacity():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001 ,
            daily_rainfall__mean_intensity=3.0 ,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2 ,
            daily_rainfall__precipitation_shape_factor=0.65,
        )


def test_a_stop_time():
    """Test that it is possible to provide a stop time"""
    mg = HexModelGrid(5, 5)

    pc = PrecipChanger(
        mg,
        daily_rainfall__intermittency_factor=0.3,
        daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
        daily_rainfall__mean_intensity=3.0,
        daily_rainfall__mean_intensity_time_rate_of_change=0.2,
        daily_rainfall__precipitation_shape_factor=0.65,
        infiltration_capacity=2.0,
        time_unit="day",
        precipchanger_start_time=10.0,
        precipchanger_stop_time=20.0,
    )

    # for the first ten steps, nothing should change
    for _ in range(10):
        pc.run_one_step(1.0)
        i, p = pc.get_current_precip_params()
        f = pc.get_erodability_adjustment_factor()
        assert i == pc.starting_frac_wet_days
        assert p == pc.starting_daily_mean_depth
        assert f == 1.0

    # run 10 more steps and save
    pc.run_one_step(10.0)
    i_end, p_end = pc.get_current_precip_params()
    f_end = pc.get_erodability_adjustment_factor()

    # then verify that no change occurs again.
    for _ in range(10):
        pc.run_one_step(1.0)
        i, p = pc.get_current_precip_params()
        f = pc.get_erodability_adjustment_factor()
        assert i == i_end
        assert p == p_end
        assert f == f_end


def test_bad_intermittency():
    """Test intermittency factors that are too big or small."""
    mg = RasterModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=-0.001,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            daily_rainfall__mean_intensity=3.0,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=0,
        )

    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=1.001,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            daily_rainfall__mean_intensity=3.0,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=0,
        )


def test_bad_intensity():
    """Test rainfall intensity that is too small."""
    mg = RasterModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=1.0,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            daily_rainfall__mean_intensity=-1,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=0,
        )


def test_bad_infiltration():
    """Test infiltration_capacity that is too small."""
    mg = RasterModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=1.0,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            daily_rainfall__mean_intensity=0.34,
            daily_rainfall__mean_intensity_time_rate_of_change=0.2,
            daily_rainfall__precipitation_shape_factor=0.65,
            infiltration_capacity=-0.001,
        )
