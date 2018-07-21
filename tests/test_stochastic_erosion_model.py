# coding: utf8
#! /usr/env/python

import pytest

from terrainbento import StochasticErosionModel, BasicSt

def test_defaults():
    params = {
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.}
    model = StochasticErosionModel(params=params)
    assert model.opt_stochastic_duration == False
    assert model.record_rain == False


def test_run_opt_true():
    params = {
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "record_rain": True
    }
    model = StochasticErosionModel(params=params)
    assert model.record_rain == True
    assert isinstance(model.rain_record, dict)
    fields = ["event_start_time", "event_duration", "rainfall_rate", "runoff_rate"]
    for f in fields:
        assert f in model.rain_record
        assert len(model.rain_record[f]) == 0


def test_run_opt_false():
    params = {
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "record_rain": False
    }
    model = StochasticErosionModel(params=params)
    assert model.record_rain == False
    assert model.rain_record is None


def test_freq_file_with_opt_duration_true():
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "random_seed": 3141,
        "frequency_filename": "yams.txt",
        "opt_stochastic_duration": True
    }
    with pytest.raises(ValueError):
        _ = StochasticErosionModel(params=params)


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


def test_write_files_no_record():
    pass
    # both of these raise value errors.
