# coding: utf8
# !/usr/env/python

import numpy as np
import pytest

from terrainbento import BasicDd, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp", [1.0 / 3, 0.5])
@pytest.mark.parametrize("n_sp", [1.0])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("threshold", [0.1])
@pytest.mark.parametrize("thresh_change_per_depth", [0.0])
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

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "water_erosion_rule__threshold": threshold,
        "water_erosion_rule__thresh_depth_derivative": thresh_change_per_depth,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicDd(**params)
    for _ in range(200):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual
    # slopes to fall.
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes_upper = (
        (U + threshold) / (K * (actual_areas ** m_sp))
    ) ** (1.0 / n_sp)
    predicted_slopes_lower = ((U + 0.0) / (K * (actual_areas ** m_sp))) ** (
        1.0 / n_sp
    )

    # assert actual and predicted slopes are in the correct range for the
    # slopes.
    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        > predicted_slopes_lower[model.grid.core_nodes[1:-1]]
    )

    assert np.all(
        actual_slopes[model.grid.core_nodes[1:-1]]
        < predicted_slopes_upper[model.grid.core_nodes[1:-1]]
    )
