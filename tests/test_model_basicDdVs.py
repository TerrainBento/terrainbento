# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicDdVs, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp", [1. / 3, 0.5])
@pytest.mark.parametrize("n_sp", [1.])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("threshold", [0.1])
@pytest.mark.parametrize("thresh_change_per_depth", [0.])
def test_steady_Ksp_no_precip_changer_no_thresh_change(
    clock_simple,
    grid_2,
    U,
    K,
    m_sp,
    n_sp,
    depression_finder,
    threshold,
    thresh_change_per_depth,
):

    step = 1000
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "water_erosion_rule__threshold": threshold,
        "water_erosion_rule__thresh_depth_derivative": thresh_change_per_depth,
        "hydraulic_conductivity": 0.,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicDdVs(**params)
    for _ in range(100):
        model.run_one_step(1000)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes_lower = ((U + 0.0) / (K * (actual_areas ** m_sp))) ** (
        1. / n_sp
    )

    # assert actual and predicted slopes are in the correct range for the
    # slopes.
    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        > predicted_slopes_lower[model.grid.core_nodes[1:-1]]
    )


def test_Aeff(clock_simple, grid_2, K, U):
    m_sp = 0.5
    n_sp = 1.0
    threshold = 0.01
    thresh_change_per_depth = 0.0
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "water_erosion_rule__threshold": threshold,
        "water_erosion_rule__thresh_depth_derivative": thresh_change_per_depth,
        "hydraulic_conductivity": 0.02,
        "recharge_rate": 1.0,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": "DepressionFinderAndRouter",
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicDdVs(**params)
    for _ in range(100):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    alpha = (
        params["hydraulic_conductivity"]
        * grid_2.at_node["soil__depth"][0]
        * grid_2.dx
        / params["recharge_rate"]
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

    # somewhat circular test to make sure slopes are below predicted upper
    # bound
    predicted_slopes_eff_upper = (
        (U + threshold) / (K * (model.eff_area ** m_sp))
    ) ** (1. / n_sp)
    predicted_slopes_eff_lower = (
        (U + 0.0) / (K * (model.eff_area ** m_sp))
    ) ** (1. / n_sp)

    # somewhat circular test to make sure VSA slopes are higher than expected
    # "normal" slopes
    predicted_slopes_normal_upper = (
        (U + threshold) / (K * (actual_areas ** m_sp))
    ) ** (1. / n_sp)
    predicted_slopes_normal_lower = (
        (U + 0.0) / (K * (actual_areas ** m_sp))
    ) ** (1. / n_sp)

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
