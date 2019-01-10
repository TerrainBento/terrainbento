# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    BasicDdVs,
    BasicHyVs,
    BasicTh,
    BasicThVs,
    BasicVs,
    NotCoreNodeBaselevelHandler,
)


@pytest.mark.parametrize("Model", [BasicVs, BasicDdVs, BasicHyVs, BasicTh])
def test_Aeff(clock_simple, grid_2, K, U, Model):
    m_sp = 0.5
    n_sp = 1.0
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "hydraulic_conductivity": 0.02,
        "recharge_rate": 1.0,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": "DepressionFinderAndRouter",
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = Model(**params)
    for _ in range(100):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]

    alpha = (
        params["hydraulic_conductivity"]
        * grid_2.at_node["soil__depth"][0]
        * grid_2.dx
        / params["recharge_rate"]
    )
    A_eff_predicted = actual_areas * np.exp(-(-alpha * actual_slopes) / actual_areas)

    # assert aeff internally calculated correclty
    assert_array_almost_equal(
        model.eff_area[model.grid.core_nodes],
        A_eff_predicted[model.grid.core_nodes],
        decimal=1,
    )

    # somewhat circular test to make sure slopes are below predicted upper
    # bound
    predicted_slopes_eff_upper = ((U) / (K * (model.eff_area ** m_sp))) ** (1. / n_sp)
    predicted_slopes_eff_lower = ((U + 0.0) / (K * (model.eff_area ** m_sp))) ** (
        1. / n_sp
    )

    # somewhat circular test to make sure VSA slopes are higher than expected
    # "normal" slopes
    predicted_slopes_normal_upper = ((U) / (K * (actual_areas ** m_sp))) ** (1. / n_sp)
    predicted_slopes_normal_lower = ((U) / (K * (actual_areas ** m_sp))) ** (1. / n_sp)

    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        > predicted_slopes_eff_lower[model.grid.core_nodes[1:-1]]
    )

    assert np.all(
        predicted_slopes_eff_lower[model.grid.core_nodes[1:-1]]
        > predicted_slopes_normal_lower[model.grid.core_nodes[1:-1]]
    )
