# coding: utf8
# !/usr/env/python

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicCh, NotCoreNodeBaselevelHandler


def test_diffusion_only():
    U = 0.0005
    K = 0.0
    m = 0.5
    n = 1.0
    D = 1.0
    S_c = 0.3
    dx = 10.0
    runtime = 30000

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_09,
        "number_of_node_rows": 3,
        "number_of_node_columns": 21,
        "node_spacing": dx,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability": K,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": S_c,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # Construct and run model
    model = BasicCh(params=params)
    for _ in range(runtime):
        model.run_one_step(clock_09["step"])

    # Construct actual and predicted slope at right edge of domain
    x = 8.5 * dx
    qs = U * x
    nterms = 11
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = D * (1 / (S_c ** (2 * (k - 1))))
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
