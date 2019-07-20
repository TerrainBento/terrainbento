# coding: utf8
# !/usr/env/python

import pytest
from scipy.integrate import IntegrationWarning

from landlab import HexModelGrid, RasterModelGrid
from terrainbento.boundary_handlers import PrecipChanger


def test_not_passing_daily_rainfall__intermittency_factor():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
            infiltration_capacity=2.0,
        )


def test_not_passing_daily_rainfall__intermittency_factor_troc():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
            infiltration_capacity=2.0,
        )


def test_not_passing_rainfall__mean_rate():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
            infiltration_capacity=2.0,
        )


def test_not_passing_rainfall__mean_rate_time_rate_of_change():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
            rainfall__mean_rate=3.0,
            rainfall__shape_factor=0.65,
            infiltration_capacity=2.0,
        )


def test_not_passing_rainfall__shape_factor():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            infiltration_capacity=2.0,
        )


def test_not_passing_infiltration_capacity():
    mg = HexModelGrid(5, 5)
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=0.3,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
        )


def test_a_stop_time():
    """Test that it is possible to provide a stop time."""
    mg = HexModelGrid(5, 5)

    pc = PrecipChanger(
        mg,
        daily_rainfall__intermittency_factor=0.3,
        daily_rainfall__intermittency_factor_time_rate_of_change=0.001,
        rainfall__mean_rate=3.0,
        rainfall__mean_rate_time_rate_of_change=0.2,
        rainfall__shape_factor=0.65,
        infiltration_capacity=2.0,
        time_unit="day",
        precipchanger_start_time=10.0,
        precipchanger_stop_time=20.0,
    )

    # for the first ten steps, nothing should change
    for _ in range(10):
        pc.run_one_step(1.0)
        i, p = pc.get_current_precip_params()
        f = pc.get_erodibility_adjustment_factor()
        assert i == pc.starting_frac_wet_days
        assert p == pc.starting_daily_mean_depth
        assert f == 1.0

    # run 10 more steps and save
    pc.run_one_step(10.0)
    i_end, p_end = pc.get_current_precip_params()
    f_end = pc.get_erodibility_adjustment_factor()

    # then verify that no change occurs again.
    for _ in range(10):
        pc.run_one_step(1.0)
        i, p = pc.get_current_precip_params()
        f = pc.get_erodibility_adjustment_factor()
        assert i == i_end
        assert p == p_end
        assert f == f_end


def test_bad_intermittency():
    """Test intermittency factors that are too big or small."""
    mg = RasterModelGrid((5, 5))
    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=-0.001,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
            infiltration_capacity=0,
        )

    with pytest.raises(ValueError):
        PrecipChanger(
            mg,
            daily_rainfall__intermittency_factor=1.001,
            daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
            rainfall__mean_rate=3.0,
            rainfall__mean_rate_time_rate_of_change=0.2,
            rainfall__shape_factor=0.65,
            infiltration_capacity=0,
        )


def test_bad_intensity():
    """Test rainfall intensity that is too small."""
    mg = RasterModelGrid((5, 5))
    with pytest.raises(ValueError):
        with pytest.warns(IntegrationWarning):
            PrecipChanger(
                mg,
                daily_rainfall__intermittency_factor=1.0,
                daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
                rainfall__mean_rate=-1,
                rainfall__mean_rate_time_rate_of_change=0.2,
                rainfall__shape_factor=0.65,
                infiltration_capacity=0,
            )


def test_bad_infiltration():
    """Test infiltration_capacity that is too small."""
    mg = RasterModelGrid((5, 5))
    with pytest.raises(ValueError):
        with pytest.warns(IntegrationWarning):
            PrecipChanger(
                mg,
                daily_rainfall__intermittency_factor=1.0,
                daily_rainfall__intermittency_factor_time_rate_of_change=0.01,
                rainfall__mean_rate=0.34,
                rainfall__mean_rate_time_rate_of_change=0.2,
                rainfall__shape_factor=0.65,
                infiltration_capacity=-0.001,
            )
