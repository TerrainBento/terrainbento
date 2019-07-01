# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    BasicDdVs,
    BasicHyVs,
    BasicVs,
    NotCoreNodeBaselevelHandler,
)


@pytest.mark.parametrize("Model", [BasicVs, BasicDdVs, BasicHyVs])
def test_Aeff(clock_simple, grid_2, K, U, Model):
    m_sp = 0.5
    n_sp = 1.0
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "hydraulic_conductivity": 0.02,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": "DepressionFinderAndRouter",
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = Model(**params)
    for _ in range(1000):
        model.run_one_step(10)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    alpha = (
        params["hydraulic_conductivity"]
        * grid_2.at_node["soil__depth"][0]
        * grid_2.dx
        / grid_2.at_node["rainfall__flux"][0]
    )

    A_eff_predicted = actual_areas * np.exp(
        -(-alpha * actual_slopes) / actual_areas
    )

    # assert aeff internally calculated correclty
    assert_array_almost_equal(
        model.grid.at_node["surface_water__discharge"][model.grid.core_nodes],
        A_eff_predicted[model.grid.core_nodes],
        decimal=1,
    )
