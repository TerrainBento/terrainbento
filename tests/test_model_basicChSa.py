# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChSa, NotCoreNodeBaselevelHandler


# test diffusion without stream power
def test_diffusion_only(clock_08, grid_4):
    U = 0.001
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    S_c = 0.2

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_4, modify_core_nodes=True, lowering_rate=-U
    )

    # Construct dictionary. Note that stream power is turned off
    params = {
        "grid": grid_4,
        "clock": clock_08,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "critical_slope": S_c,
        "water_erodibility": 0,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # Construct and run model
    model = BasicChSa(**params)
    for _ in range(30000):
        model.run_one_step(clock_08.step)

    # test steady state soil depth
    actual_depth = model.grid.at_node["soil__depth"][30]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=2)

    # Construct actual and predicted slope at right edge of domain
    x = 8.5 * grid_4.dx
    qs = U * x
    nterms = 11
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = (
            regolith_transport_parameter
            * soil_transport_decay_depth
            * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
            * (1 / (S_c ** (2 * (k - 1))))
        )
    p = np.fliplr([p])[0]
    p = np.append(p, qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    actual_slope = np.abs(
        model.grid.at_node["topographic__steepest_slope"][39]
    )
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=1)
