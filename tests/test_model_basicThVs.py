# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicThVs
from terrainbento.utilities import filecmp


def test_Aeff():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 1000
    threshold = 0.01
    hydraulic_conductivity = 0.1
    soil__initial_thickness = 0.1
    node_spacing = 100.0
    recharge_rate = 0.5
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
        "hydraulic_conductivity": 0.1,
        "soil__initial_thickness": 0.1,
        "recharge_rate": 0.5,
        "m_sp": m,
        "n_sp": n,
        "water_erosion_rule__threshold": threshold,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    model = BasicThVs(params=params)
    for _ in range(200):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    alpha = (
        hydraulic_conductivity
        * soil__initial_thickness
        * node_spacing
        / recharge_rate
    )
    A_eff_predicted = actual_areas * np.exp(
        -(-alpha * actual_slopes) / actual_areas
    )

    # assert aeff internally calculated correclty
    assert_array_almost_equal(
        model.eff_area[model.grid.core_nodes],
        A_eff_predicted[model.grid.core_nodes],
        decimal=1,
    )

    # somewhat circular test to make sure slopes are below predicted upper bound
    predicted_slopes_eff_upper = (
        (U + threshold) / (K * (model.eff_area ** m))
    ) ** (1. / n)

    # somewhat circular test to make sure slopes are below predicted upper
    # bound
    predicted_slopes_eff_upper = (
        (U + threshold) / (K * (model.eff_area ** m))
    ) ** (1. / n)
    predicted_slopes_eff_lower = ((U + 0.0) / (K * (model.eff_area ** m))) ** (
        1. / n
    )

    # somewhat circular test to make sure VSA slopes are higher than expected
    # "normal" slopes
    predicted_slopes_normal_upper = (
        (U + threshold) / (K * (actual_areas ** m))
    ) ** (1. / n)
    predicted_slopes_normal_lower = (
        (U + 0.0) / (K * (actual_areas ** m))
    ) ** (1. / n)

    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        < predicted_slopes_eff_upper[model.grid.core_nodes[1:-1]]
    )
    assert np.all(
        predicted_slopes_eff_upper[model.grid.core_nodes[1:-1]]
        > predicted_slopes_normal_upper[model.grid.core_nodes[1:-1]]
    )

    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        > predicted_slopes_eff_lower[model.grid.core_nodes[1:-1]]
    )

    assert np.all(
        predicted_slopes_eff_lower[model.grid.core_nodes[1:-1]]
        > predicted_slopes_normal_lower[model.grid.core_nodes[1:-1]]
    )


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 1000
    threshold = 0.01
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.1,
        "recharge_rate": 0.5,
        "m_sp": m,
        "n_sp": n,
        "water_erosion_rule__threshold": threshold,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicThVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    # construct actual and predicted slopes
    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual
    # slopes to fall.
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes_upper = ((U + threshold) / (K * (actual_areas ** m))) ** (
        1. / n
    )
    predicted_slopes_lower = ((U + 0.0) / (K * (actual_areas ** m))) ** (
        1. / n
    )

    # assert actual and predicted slopes are in the correct range for the
    # slopes.
    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        > predicted_slopes_lower[model.grid.core_nodes[1:-1]]
    )

    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        < predicted_slopes_upper[model.grid.core_nodes[1:-1]]
    )


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    K = 0.001
    m = 0.5
    n = 1.0
    step = 1000
    threshold = 0.000001
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.1,
        "recharge_rate": 0.5,
        "m_sp": m,
        "n_sp": n,
        "water_erosion_rule__threshold": threshold,
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicThVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = ((U / K + threshold) / ((actual_areas ** m))) ** (
        1. / n
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )
