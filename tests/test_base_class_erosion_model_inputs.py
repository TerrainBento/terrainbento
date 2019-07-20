# coding: utf8
# !/usr/env/python

import pytest

from landlab import HexModelGrid, RasterModelGrid
from landlab.components import FlowAccumulator
from terrainbento import ErosionModel

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
    "water__unit_flux_in",
    "rainfall__flux",
]


def test_not_correct_fields(clock_simple):
    grid = RasterModelGrid((3, 21))
    with pytest.raises(ValueError):
        ErosionModel(clock=clock_simple, grid=grid)


def test_no_clock(simple_square_grid):
    with pytest.raises(ValueError):
        ErosionModel(clock="spam", grid=simple_square_grid)


def test_no_grid(clock_simple):
    with pytest.raises(ValueError):
        ErosionModel(grid="eggs", clock=clock_simple)


def test_no_clock_in_file(tmpdir, basic_inputs_no_clock_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_no_clock_yaml)
        with pytest.raises(ValueError):
            ErosionModel.from_file("./params.yaml")


def test_no_grid_in_file(tmpdir, basic_inputs_no_grid_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_no_grid_yaml)
        with pytest.raises(ValueError):
            ErosionModel.from_file("./params.yaml")


def test_input_file(tmpdir, inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(inputs_yaml)

        em = ErosionModel.from_file("./params.yaml")

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
    assert em._model_time == 0.0


def test_parameters(clock_simple):
    params = {
        "grid": {
            "HexModelGrid": [
                {"base_num_rows": 8, "base_num_cols": 5, "dx": 10},
                {
                    "fields": {
                        "node": {
                            "topographic__elevation": {
                                "constant": [{"value": 0}]
                            }
                        }
                    }
                },
            ]
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
    assert em._model_time == 0.0


def test_string(tmpdir, inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(inputs_yaml)

        with open("./params.yaml", "r") as f:
            contents = f.read()

    em = ErosionModel.from_file(contents)
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
    assert em._model_time == 0.0


def test_string_D8(tmpdir, inputs_D8_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(inputs_D8_yaml)

        with open("./params.yaml", "r") as f:
            contents = f.read()

    em = ErosionModel.from_file(contents)
    assert isinstance(em.grid, RasterModelGrid)
    assert em.grid.number_of_nodes == 20
    for field in at_node_fields:
        assert field in em.grid.at_node
    assert isinstance(em.flow_accumulator, FlowAccumulator) is True
    assert em.flow_accumulator.flow_director._name == "FlowDirectorD8"
    assert em.boundary_handlers == {}
    assert em.output_writers == {}
    assert em.save_first_timestep is True
    assert em._out_file_name == "terrainbento_output"
    assert em._model_time == 0.0
