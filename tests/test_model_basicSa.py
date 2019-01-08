# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento.utilities import filecmp


# test diffusion without stream power
def test_diffusion_only():
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    step = 10
    dx = 10.0
    number_of_node_columns = 21
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    runtime = 100000

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"step": step, "output_interval": 2., "stop": 200.},
        "number_of_node_rows": 3,
        "number_of_node_columns": 21,
        "node_spacing": dx,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": initial_soil_thickness,
        "water_erodability": K,
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
    model = BasicSa(params=params)
    for _ in range(runtime):
        model.run_one_step(step)

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
