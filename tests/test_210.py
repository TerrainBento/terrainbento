import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal  # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicHyVs

def test_Aeff
    U = 0.0001
    K = 0.001
    m = 1. / 3.
    n = 2. / 3.
    dt = 1000
    hydraulic_conductivity = 0.1
    soil__initial_thickness = 0.1
    recharge_rate = 0.5
    node_spacing = 100.0

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
        "hydraulic_conductivity": hydraulic_conductivity,
        "soil__initial_thickness": soil__initial_thickness,
        "recharge_rate": recharge_rate,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    model = BasicHyVs(params=params)
    for i in range(200):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    alpha = hydraulic_conductivity * soil__initial_thickness*node_spacing/recharge_rate
    A_eff_predicted = actual_areas*np.exp(-(-alpha*actual_slopes)/actual_areas)

    # assert aeff internally calculated correclty
    assert_array_almost_equal(model.eff_area[model.grid.core_nodes], A_eff_predicted[model.grid.core_nodes], decimal = 2)

    # assert correct s a relationship (slightly circular)
    predicted_slopes = (U / (K * (A_eff_predicted ** m))) ** (1. / n)
    assert_array_equal(actual_slopes[model.grid.core_nodes], predicted_slopes[model.grid.core_nodes])

    # assert all slopes above non effective
    predicted_slopes_normal = (U / (K * (actual_areas ** m))) ** (1. / n)
    assert np.all(actual_slopes[model.grid.core_nodes]>predicted_slopes_normal[model.grid.core_nodes]) == True





def test_steady_Kss_no_precip_changer():
    U = 0.0001
    K = 0.003
    m = 1. / 3.
    n = 1.
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0

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
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "solver": "basic",
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHyVs(params=params)
    for i in range(2000):
        model.run_one_step(dt)

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
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHyVs(params=params)
    for i in range(800):
        model.run_one_step(dt)

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
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHyVs(params=params)
    for i in range(800):
        model.run_one_step(dt)

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
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHyVs(params=params)
    for i in range(800):
        model.run_one_step(dt)

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


def test_with_precip_changer():
    K = 0.01
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
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
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": {
            "daily_rainfall__intermittency_factor": 0.5,
            "daily_rainfall__intermittency_factor_time_rate_of_change": 0.1,
            "daily_rainfall__mean_intensity": 1.0,
            "daily_rainfall__mean_intensity_time_rate_of_change": 0.2,
        },
    }

    model = BasicHyVs(params=params)
    assert model.eroder.K[0] == K
    assert "PrecipChanger" in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K, 5) == 0.10326


def test_diffusion_only():
    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.75
    n = 1.0
    dt = 1000
    v_sc = 0.001
    phi = 0.1
    F_f = 0.1

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 21,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "west_boundary_closed": False,
        "south_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability": 0.0,
        "hydraulic_conductivity": 0.0,
        "soil__initial_thickness": 0.0,
        "recharge_rate": 0.5,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }
    nts = int(total_time / dt)

    reference_node = 9
    # construct and run model
    model = BasicHyVs(params=params)
    for i in range(nts):
        model.run_one_step(dt)

    predicted_z = model.z[model.grid.core_nodes[reference_node]] - (U / (2. * D)) * (
        (
            model.grid.x_of_node
            - model.grid.x_of_node[model.grid.core_nodes[reference_node]]
        )
        ** 2
    )

    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(
        predicted_z[model.grid.core_nodes], model.z[model.grid.core_nodes], decimal=2
    )

