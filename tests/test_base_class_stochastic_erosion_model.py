# coding: utf8
# !/usr/env/python
import os

import numpy as np
import pytest

from terrainbento import BasicSt, PrecipChanger, StochasticErosionModel
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_defaults(clock_simple, grid_1):
    model = StochasticErosionModel(clock=clock_simple, grid=grid_1)
    assert model.opt_stochastic_duration is False
    assert model.record_rain is False


def test_init_record_opt_true(clock_simple, grid_1):
    model = StochasticErosionModel(
        clock=clock_simple, grid=grid_1, record_rain=True
    )
    assert model.record_rain is True
    assert isinstance(model.rain_record, dict)
    fields = [
        "event_start_time",
        "event_duration",
        "rainfall_rate",
        "runoff_rate",
    ]
    for f in fields:
        assert f in model.rain_record
        assert len(model.rain_record[f]) == 0


def test_init_record_opt_false(clock_simple, grid_1):
    params = {"clock": clock_simple, "record_rain": False, "grid": grid_1}
    model = StochasticErosionModel(**params)
    assert model.record_rain is False
    assert model.rain_record is None


def test_run_stochastic_opt_true(clock_04, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": True,
        "clock": clock_04,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "mean_storm_duration": 2.0,
        "mean_interstorm_duration": 3.0,
        "mean_storm_depth": 1.0,
        "random_seed": 1234,
    }

    model = BasicSt(**params)
    assert model.opt_stochastic_duration is True
    model.run_for(model.clock.step, model.clock.stop)

    rainfall_rate = np.asarray(model.rain_record["rainfall_rate"]).round(
        decimals=5
    )
    event_duration = np.asarray(model.rain_record["event_duration"]).round(
        decimals=5
    )

    dry_times = event_duration[rainfall_rate == 0]
    wet_times = event_duration[rainfall_rate > 0]

    np.testing.assert_almost_equal(
        np.round(np.mean(dry_times), decimals=1),
        params["mean_interstorm_duration"],
        decimal=1,
    )
    np.testing.assert_almost_equal(
        np.round(np.mean(wet_times), decimals=1),
        params["mean_storm_duration"],
        decimal=1,
    )

    avg_storm_depth = np.sum((rainfall_rate * event_duration)) / len(wet_times)

    np.testing.assert_array_almost_equal(
        avg_storm_depth, params["mean_storm_depth"], decimal=1
    )


def test_run_stochastic_opt_false(clock_05, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_05,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }

    model = BasicSt(**params)
    assert model.opt_stochastic_duration is False
    model.run_for(model.clock.step, 10000.0)

    rainfall_rate = np.asarray(model.rain_record["rainfall_rate"])
    event_duration = np.asarray(model.rain_record["event_duration"])

    dry_times = event_duration[rainfall_rate == 0]
    wet_times = event_duration[rainfall_rate > 0]

    assert (
        np.array_equiv(
            dry_times,
            model.clock.step * (1.0 - params["rainfall_intermittency_factor"]),
        )
        is True
    )
    assert (
        np.array_equiv(
            wet_times,
            model.clock.step * (params["rainfall_intermittency_factor"]),
        )
        is True
    )

    avg_storm_depth = np.sum((rainfall_rate * event_duration)) / len(wet_times)

    np.testing.assert_array_almost_equal(
        avg_storm_depth, params["rainfall__mean_rate"], decimal=1
    )


def test_reset_random_seed_stochastic_duration_true(clock_simple, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": True,
        "clock": clock_simple,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "mean_storm_duration": 2.0,
        "mean_interstorm_duration": 3.0,
        "mean_storm_depth": 1.0,
        "random_seed": 0,
    }

    model = BasicSt(**params)
    step = 1
    runtime = 200

    model.rain_generator.delta_t = step
    model.rain_generator.run_time = runtime
    model.reset_random_seed()
    duration_1 = []
    precip_1 = []

    for (
        tr,
        p,
    ) in model.rain_generator.yield_storm_interstorm_duration_intensity():
        precip_1.append(p)
        duration_1.append(tr)

    model.rain_generator.delta_t = step
    model.rain_generator.run_time = runtime
    model.reset_random_seed()

    duration_2 = []
    precip_2 = []

    for (
        tr,
        p,
    ) in model.rain_generator.yield_storm_interstorm_duration_intensity():
        precip_2.append(p)
        duration_2.append(tr)

    np.testing.assert_array_equal(duration_1, duration_2)
    np.testing.assert_array_equal(precip_1, precip_2)


def test_reset_random_seed_stochastic_duration_false(clock_05, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_05,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }
    model = BasicSt(**params)

    model.reset_random_seed()
    depth_1 = []
    for _ in range(10):
        depth_1.append(
            model.rain_generator.generate_from_stretched_exponential(
                model.scale_factor, model.shape_factor
            )
        )

    model.reset_random_seed()
    depth_2 = []
    for _ in range(10):
        depth_2.append(
            model.rain_generator.generate_from_stretched_exponential(
                model.scale_factor, model.shape_factor
            )
        )
    np.testing.assert_array_equal(depth_1, depth_2)


def test_float_number_of_sub_time_steps(clock_05, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_05,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1.5,
        "random_seed": 1234,
    }
    with pytest.raises(ValueError):
        BasicSt(**params)


def test_run_opt_false_with_changer(clock_06, grid_1, precip_defaults):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_06,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.5,
        "rainfall__shape_factor": 0.65,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }

    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    assert "PrecipChanger" in model.boundary_handlers

    predicted_intermittency = params[
        "rainfall_intermittency_factor"
    ] + precip_defaults[
        "daily_rainfall__intermittency_factor_time_rate_of_change"
    ] * (
        model.clock.stop - model.clock.step
    )

    predicted_intensity = params["rainfall__mean_rate"] + precip_defaults[
        "rainfall__mean_rate_time_rate_of_change"
    ] * (model.clock.stop - model.clock.step)

    assert model.rainfall_intermittency_factor == predicted_intermittency
    assert model.rainfall__mean_rate == predicted_intensity


def test_opt_dur_true_with_changer(clock_02, grid_1, precip_defaults):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": True,
        "clock": clock_02,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }

    with pytest.raises(ValueError):
        StochasticErosionModel(**params)


def test_not_specifying_record_rain(clock_05, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_05,
        "record_rain": False,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }

    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    with pytest.raises(ValueError):
        model.write_storm_sequence_to_file()

    with pytest.raises(ValueError):
        model.write_exceedance_frequency_file()


def test_finalize_opt_duration_stochastic_false_too_short(clock_05, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_05,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }

    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    with pytest.raises(RuntimeError):
        model.finalize()

    os.remove("storm_sequence.txt")


def test_finalize_opt_duration_stochastic_false_no_rain(clock_07, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_07,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.0,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }
    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    with pytest.raises(ValueError):
        model.finalize()


def test_finalize_opt_duration_stochastic_false(clock_07, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_07,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }
    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    model.finalize()

    # assert that these are correct
    truth_file = os.path.join(
        _TEST_DATA_DIR, "opt_dur_false_storm_sequence.txt"
    )
    assert filecmp("storm_sequence.txt", truth_file) is True

    truth_file = os.path.join(
        _TEST_DATA_DIR, "opt_dur_false_exceedance_summary.txt"
    )
    assert filecmp("exceedance_summary.txt", truth_file) is True

    os.remove("storm_sequence.txt")
    os.remove("exceedance_summary.txt")


def test_finalize_opt_duration_stochastic_true(clock_07, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": True,
        "clock": clock_07,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "mean_storm_duration": 2.0,
        "mean_interstorm_duration": 3.0,
        "mean_storm_depth": 1.0,
        "random_seed": 1234,
    }

    model = BasicSt(**params)
    model.reset_random_seed()
    model.run_for(model.clock.step, model.clock.stop)
    model.finalize()

    # assert that these are correct
    truth_file = os.path.join(
        _TEST_DATA_DIR, "opt_dur_true_storm_sequence.txt"
    )
    assert filecmp("storm_sequence.txt", truth_file) is True

    os.remove("storm_sequence.txt")


def test_runoff_equals_zero(clock_07, grid_1):
    params = {
        "grid": grid_1,
        "opt_stochastic_duration": False,
        "clock": clock_07,
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodibility": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 100000.0,
        "rainfall__mean_rate": 0.0,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 1.0,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
    }
    model = BasicSt(**params)
    model.run_one_step(1.0)
    runoff = model.calc_runoff_and_discharge()
    assert runoff == 0
