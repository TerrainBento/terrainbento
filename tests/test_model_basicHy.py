# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicHy
from terrainbento.utilities import filecmp


def test_steady_Kss_no_precip_changer():
    U = 0.0001
    K = 0.003
    m = 1. / 3.
    n = 1.
    step = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHy(params=params)
    for _ in range(2000):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m)))
        + (U / (K * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHy(params=params)
    for _ in range(800):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m)))
        + (U / (K * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_steady_Ksp_no_precip_changer_no_solver_given():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHy(params=params)
    for _ in range(800):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m)))
        + (U / (K * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHy(params=params)
    for _ in range(800):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m)))
        + (U / (K * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )
