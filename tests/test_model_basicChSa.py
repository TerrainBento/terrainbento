# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicChSa
from terrainbento.utilities import filecmp


# test diffusion without stream power
def test_diffusion_only(clock_08):
    U = 0.001
    K = 0.0
    m = 0.5
    n = 1.0
    dx = 10.0
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    S_c = 0.2
    runtime = 50000

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_08,
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
        "critical_slope": S_c,
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
    model = BasicChSa(params=params)
    for _ in range(runtime):
        model.run_one_step(clock_08["step"])

    # test steady state soil depth
    actual_depth = model.grid.at_node["soil__depth"][30]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=3)

    # Construct actual and predicted slope at right edge of domain
    x = 8.5 * dx
    qs = U * x
    nterms = 11
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = (
            regolith_transport_parameter
            * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
            * (1 / (S_c ** (2 * (k - 1))))
        )
    p = np.fliplr([p])[0]
    p = np.append(p, qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    # print(predicted_slope)

    actual_slope = np.abs(
        model.grid.at_node["topographic__steepest_slope"][39]
    )
    # print model.grid.at_node["topographic__steepest_slope"]
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=3)
