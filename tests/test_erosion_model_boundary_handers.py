# coding: utf8
#! /usr/env/python

import numpy as np
from numpy.testing import assert_array_equal  # , assert_array_almost_equal
import pytest

from landlab import HexModelGrid, RasterModelGrid
from landlab.components import LinearDiffuser, NormalFault

from terrainbento import ErosionModel, Basic
from terrainbento.boundary_condition_handlers import (
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
    CaptureNodeBaselevelHandler,
)


def test_bad_boundary_condition_string():
    params = {"dt": 1, "output_interval": 2., "run_duration": 10., 'BoundaryHandlers':"spam"}
    with pytest.raises(ValueError):
        ErosionModel(params=params)

def test_boundary_condition_handler_with_special_part_of_params():
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }
    model = Basic(params=params)
    bh = model.boundary_handler["NotCoreNodeBaselevelHandler"]

    # assertion tests
    assert "NotCoreNodeBaselevelHandler" in model.boundary_handler
    assert bh.lowering_rate == -U
    assert bh.prefactor == -1
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)


def test_boundary_condition_handler_without_special_part_of_params():
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    dt = 1000
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "modify_core_nodes": True,
        "lowering_rate": -U,
    }

    model = Basic(params=params)
    bh = model.boundary_handler["NotCoreNodeBaselevelHandler"]

    # assertion tests
    assert "NotCoreNodeBaselevelHandler" in model.boundary_handler
    assert bh.lowering_rate == -U
    assert bh.prefactor == -1
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)


def test_pass_two_boundary_handlers():
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": [
            "NotCoreNodeBaselevelHandler",
            "SingleNodeBaselevelHandler",
        ],
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
        "SingleNodeBaselevelHandler": {"lowering_rate": -U},
    }
    model = Basic(params=params)
    model.run_one_step(1.0)

    truth = np.zeros(model.z.size)
    truth[0] -= U
    truth[model.grid.core_nodes] += U
    assert_array_equal(model.z, truth)
