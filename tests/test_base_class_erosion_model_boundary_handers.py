# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_equal  # , assert_array_almost_equal

from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY, RasterModelGrid
from terrainbento import Basic, BasicSt, ErosionModel
from terrainbento.boundary_handlers import (
    CaptureNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
)
from terrainbento.utilities import filecmp


def test_bad_boundary_condition_string(clock_01, almost_default_grid):
    params = {"grid": almost_default_grid, "clock": clock_01, "boundary_handlers": {"spam": None}}
    with pytest.raises(ValueError):
        ErosionModel(**params)


def test_single_node_blh_with_closed_boundaries(clock_simple):
    params = {
        "clock": clock_simple,
        "number_of_node_rows": 10,
        "number_of_node_columns": 10,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "node_spacing": 10.0,  # meters
        "random_seed": 4897,  # set to initialize the topography with reproducible random noise
        "water_erodability": 0.0001,  # years^-1
        "m_sp": 0.5,  # unitless
        "n_sp": 1.0,  # unitless
        "regolith_transport_parameter": 0.01,  # meters^2/year
        "BoundaryHandlers": "SingleNodeBaselevelHandler",
        "SingleNodeBaselevelHandler": {
            "modify_outlet_node": False,
            "lowering_rate": -0.0005,
            "outlet_id": 3,
        },  # meters/year
    }
    model = Basic(**params)
    assert model.grid.status_at_node[3] == FIXED_VALUE_BOUNDARY


def test_boundary_condition_handler_without_special_part_of_params(
    clock_simple
):
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
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

    model = Basic(**params)
    bh = model.boundary_handler["NotCoreNodeBaselevelHandler"]

    # assertion tests
    assert "NotCoreNodeBaselevelHandler" in model.boundary_handler
    assert bh.lowering_rate == -U
    assert bh.prefactor == -1
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)


def test_pass_two_boundary_handlers(clock_simple):
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": [
            "NotCoreNodeBaselevelHandler",
            "SingleNodeBaselevelHandler",
        ],
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
        "SingleNodeBaselevelHandler": {"lowering_rate": -U},
    }
    model = Basic(**params)
    model.run_one_step(1.0)

    truth = np.zeros(model.z.size)
    truth[0] -= U
    truth[model.grid.core_nodes] += U
    assert_array_equal(model.z, truth)

    status_at_node = np.zeros(model.z.size)
    status_at_node[model.grid.boundary_nodes] = CLOSED_BOUNDARY
    status_at_node[0] = FIXED_VALUE_BOUNDARY
    assert_array_equal(model.grid.status_at_node, status_at_node)


def test_generic_bch(clock_simple):
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
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
        "BoundaryHandlers": "GenericFuncBaselevelHandler",
        "GenericFuncBaselevelHandler": {
            "modify_core_nodes": True,
            "function": lambda grid, t: -(
                grid.x_of_node + grid.y_of_node + (0 * t)
            ),
        },  # returns a rate in meters/year
    }
    model = Basic(**params)
    bh = model.boundary_handler["GenericFuncBaselevelHandler"]

    # assertion tests
    assert "GenericFuncBaselevelHandler" in model.boundary_handler
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)

    step = 10.
    model.run_one_step(step)

    dzdt = -(model.grid.x_of_node + model.grid.y_of_node)
    truth_z = -1. * dzdt * step
    assert_array_equal(
        model.z[model.grid.core_nodes], truth_z[model.grid.core_nodes]
    )


def test_capture_node(clock_simple):
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
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
        "BoundaryHandlers": "CaptureNodeBaselevelHandler",
        "CaptureNodeBaselevelHandler": {
            "capture_node": 1,
            "capture_incision_rate": -3.0,
            "capture_start_time": 10,
            "capture_stop_time": 20,
            "post_capture_incision_rate": -0.1,
        },  # returns a rate in meters/year
    }

    model = Basic(**params)
    # assertion tests
    assert "CaptureNodeBaselevelHandler" in model.boundary_handler
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == 0
    model.run_one_step(10.)
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == 0
    model.run_one_step(10)
    assert (
        model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == -30.
    )
    model.run_one_step(10)
    assert (
        model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == -31.
    )
