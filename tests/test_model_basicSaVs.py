# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicSaVs
from terrainbento.utilities import filecmp


# test diffusion without stream power
def test_diffusion_only():
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    dt = 10
    dx = 10.0

    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5
    number_of_node_columns = 21
    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": dt, "output_interval": 2., "run_duration": 200.},
        "number_of_node_rows": 3,
        "number_of_node_columns": number_of_node_columns,
        "node_spacing": dx,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": initial_soil_thickness,
        "water_erodability": K,
        "recharge_rate": recharge_rate,
        "hydraulic_conductivity": hydraulic_conductivity,
        "m_sp": m,
        "n_sp": n,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # Construct and run model
    model = BasicSaVs(params=params)
    for _ in range(100000):
        model.run_one_step(dt)

    # test steady state soil depth
    actual_depth = model.grid.at_node["soil__depth"][30]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=3)

    # test steady state slope
    actual_profile = model.grid.at_node["topographic__elevation"][21:42]

    domain = np.arange(0, max(model.grid.node_x + dx), dx)
    steady_domain = np.arange(-max(domain) / 2., max(domain) / 2. + dx, dx)

    half_space = int(len(domain) / 2)
    steady_z_profile_firsthalf = (steady_domain[0:half_space]) ** 2 * U / (
        regolith_transport_parameter
        * 2
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) - (U * (number_of_node_columns / 2) ** 2) / (
        2
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )
    steady_z_profile_secondhalf = -(steady_domain[half_space:]) ** 2 * U / (
        regolith_transport_parameter
        * 2
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) + (U * (number_of_node_columns / 2) ** 2) / (
        2
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )

    steady_z_profile = np.append(
        [-steady_z_profile_firsthalf], [steady_z_profile_secondhalf]
    )
    predicted_profile = steady_z_profile - np.min(steady_z_profile)

    assert_array_almost_equal(actual_profile, predicted_profile)


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.001
    K = 0.01
    m = 0.5
    n = 1.0
    dt = 1000.
    dx = 100.0
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": dt, "output_interval": 2., "run_duration": 200.},
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": dx,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": 0.0,
        "water_erodability": K,
        "hydraulic_conductivity": hydraulic_conductivity,
        "recharge_rate": recharge_rate,
        "m_sp": m,
        "n_sp": n,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicSaVs(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = (U / (K * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_steady_Ksp_no_precip_changer():
    U = 0.001
    K = 0.01
    m = 0.5
    n = 1.0
    dt = 1000.
    dx = 100.0
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"dt": dt, "output_interval": 2., "run_duration": 200.},
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": dx,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": 0.0,
        "water_erodability": K,
        "hydraulic_conductivity": hydraulic_conductivity,
        "recharge_rate": recharge_rate,
        "m_sp": m,
        "n_sp": n,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicSaVs(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = (U / (K * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_with_precip_changer():
    K = 0.01
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    soil_transport_decay_depth = 0.5
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": 0.0,
        "critical_slope": 0.2,
        "water_erodability": K,
        "hydraulic_conductivity": hydraulic_conductivity,
        "recharge_rate": recharge_rate,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults,
    }

    model = BasicSaVs(params=params)
    assert np.array_equiv(model.eroder._K_unit_time, K) is True
    assert "PrecipChanger" in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)

    truth = K * precip_testing_factor * np.ones(model.eroder._K_unit_time.size)
    assert_array_almost_equal(model.eroder._K_unit_time, truth, decimal=4)
