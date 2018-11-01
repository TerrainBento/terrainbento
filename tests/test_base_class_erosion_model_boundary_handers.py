# coding: utf8
#! /usr/env/python

import numpy as np
from numpy.testing import assert_array_equal  # , assert_array_almost_equal
import pytest

from terrainbento import ErosionModel, Basic, BasicSt
from terrainbento.boundary_condition_handlers import (
    PrecipChanger,
    SingleNodeBaselevelHandler,
    CaptureNodeBaselevelHandler,
)


def test_bad_boundary_condition_string():
    params = {
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 10.},
        "BoundaryHandlers": "spam",
    }
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
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 200.},
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


def test_boundary_condition_handler_with_bad_special_part_of_params():
    params = {
        "opt_stochastic_duration": False,
        "clock": {"dt": 10,
        "output_interval": 2.,
        "run_duration": 1000.},
        "record_rain": True,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "water_erodability~stochastic": 0.01,
        "regolith_transport_parameter": 0.1,
        "infiltration_capacity": 0.0,
        "rainfall__mean_rate": 1.2,
        "rainfall_intermittency_factor": 0.1,
        "rainfall__shape_factor": 0.6,
        "number_of_sub_time_steps": 1,
        "random_seed": 1234,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": {
            "daily_rainfall__intermittency_factor": 0.1,
            "daily_rainfall__intermittency_factor_time_rate_of_change": 0.0001,
            "rainfall__mean_rate": 1.,
            "rainfall__mean_rate_time_rate_of_change": 0.0001,
            "infiltration_capacity": 0,
            "rainfall__shape_factor": 0.65,
        },
    }
    with pytest.raises(ValueError):
        BasicSt(params=params)


def test_boundary_condition_handler_without_special_part_of_params():
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 200.},
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
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 200.},
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


def test_generic_bch():
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 200.},
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
            "function": lambda grid, t: -(grid.x_of_node + grid.y_of_node + (0 * t)),
        },  # returns a rate in meters/year
    }
    model = Basic(params=params)
    bh = model.boundary_handler["GenericFuncBaselevelHandler"]

    # assertion tests
    assert "GenericFuncBaselevelHandler" in model.boundary_handler
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)

    dt = 10.
    model.run_one_step(dt)

    dzdt = -(model.grid.x_of_node + model.grid.y_of_node)
    truth_z = -1. * dzdt * dt
    assert_array_equal(model.z[model.grid.core_nodes], truth_z[model.grid.core_nodes])


def test_capture_node():
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": 1,
        "output_interval": 2.,
        "run_duration": 200.},
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

    model = Basic(params=params)
    # assertion tests
    assert "CaptureNodeBaselevelHandler" in model.boundary_handler
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == 0
    model.run_one_step(10.)
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == 0
    model.run_one_step(10)
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == -30.
    model.run_one_step(10)
    assert model.z[params["CaptureNodeBaselevelHandler"]["capture_node"]] == -31.
