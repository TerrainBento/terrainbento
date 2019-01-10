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


def test_input_file():
    fp = os.path.join(_TEST_DATA_DIR, "inputs.yaml")
    em = ErosionModel.from_file(fp)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert isinstance(em.flow_accumulator, FlowAccumulator) is True
    assert em.flow_accumulator.flow_director._name == "FlowDirectorSteepest"
    assert em.boundary_handlers == {}
    assert em.output_writers == {}
    assert em.save_first_timestep is True
    assert em._out_file_name == "terrainbento_output"
    assert em._model_time == 0.


def test_parameters(clock_simple):
    params = {
        "grid": {
            "grid": {
                "HexModelGrid": {"base_num_rows": 8, "base_num_cols": 5, "dx": 10}
            },
            "fields": {
                "at_node": {"topographic__elevation": {"constant": {"constant": 0}}}
            },
        },
        "clock": {"step": 1, "stop": 10},
        "output_interval": 2,
    }

    em = ErosionModel.from_dict(params)
    assert isinstance(em.grid, HexModelGrid)
    assert em.grid.number_of_nodes == 56
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert isinstance(em.flow_accumulator, FlowAccumulator) is True
    assert em.flow_accumulator.flow_director._name == "FlowDirectorSteepest"
    assert em.boundary_handlers == {}
    assert em.output_writers == {}
    assert em.save_first_timestep is True
    assert em._out_file_name == "terrainbento_output"
    assert em._model_time == 0.
