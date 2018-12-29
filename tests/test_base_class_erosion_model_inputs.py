# coding: utf8
# !/usr/env/python

import os

import pytest

from landlab import HexModelGrid
from landlab.components import FlowAccumulator
from terrainbento import ErosionModel
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

at_node_fields = [
    "topographic__elevation",
    "initial_topographic__elevation",
    "cumulative_elevation_change",
    "water__unit_flux_in",
    "flow__receiver_node",
    "topographic__steepest_slope",
    "flow__link_to_receiver_node",
    "flow__sink_flag",
    "drainage_area",
    "surface_water__discharge",
    "flow__upstream_node_order",
    "flow__data_structure_delta",
]


def test_no_inputs():
    with pytest.raises(ValueError):
        ErosionModel()


def test_both_inputs():
    params = {"model_grid": "HexModelGrid", "clock": clock_01}
    fp = os.path.join(_TEST_DATA_DIR, "inputs.txt")

    with pytest.raises(ValueError):
        ErosionModel(params=params, input_file=fp)


def test_both_node_rows_and_DEM():
    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": 1, "output_interval": 2., "run_duration": 100},
        "number_of_node_rows": 5,
        "DEM_filename": "foo.nc",
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)


def test_no_required_params():
    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": 1, "output_interval": 2.},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)

    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": 1, "run_duration": 10.},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)

    params = {
        "model_grid": "HexModelGrid",
        "clock": {"output_interval": 2, "run_duration": 10.},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)


def test_bad_req_params():
    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": "spam", "output_interval": 2., "run_duration": 10.},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)

    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": 1, "output_interval": "eggs", "run_duration": 10.},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)

    params = {
        "model_grid": "HexModelGrid",
        "clock": {"dt": 1, "output_interval": 2., "run_duration": "wooo"},
    }
    with pytest.raises(ValueError):
        ErosionModel(params=params)


def test_input_file():
    fp = os.path.join(_TEST_DATA_DIR, "inputs.yaml")
    em = ErosionModel(input_file=fp)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert em.flow_director == "FlowDirectorSteepest"
    assert isinstance(em.flow_accumulator, FlowAccumulator) is True
    assert em.depression_finder is None
    assert em.boundary_handler == {}
    assert em.output_writers == {"class": {}, "function": []}
    assert em.save_first_timestep is True
    assert em._out_file_name == "terrainbento_output"
    assert em._model_time == 0.


def test_parameters():
    params = {"model_grid": "HexModelGrid", "clock": clock_01}
    em = ErosionModel(params=params)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert em.flow_director == "FlowDirectorSteepest"
    assert isinstance(em.flow_accumulator, FlowAccumulator) is True
    assert em.depression_finder is None
    assert em.boundary_handler == {}
    assert em.output_writers == {"class": {}, "function": []}
    assert em.save_first_timestep is True
    assert em._out_file_name == "terrainbento_output"
    assert em._model_time == 0.
