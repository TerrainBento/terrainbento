# coding: utf8
# !/usr/env/python

import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import Basic, NotCoreNodeBaselevelHandler, PrecipChanger


def test_diffusion_only(clock_simple, grid_1, U):
    total_time = 5.0e6
    step = 1000
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 1,
        "water_erodability": 0,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    # construct and run model
    model = Basic(**params)

    nts = int(total_time / step)
    for _ in range(nts):
        model.run_one_step(1000)
    reference_node = 9
    predicted_z = model.z[model.grid.core_nodes[reference_node]] - (
        U / (2. * params["regolith_transport_parameter"])
    ) * (
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
