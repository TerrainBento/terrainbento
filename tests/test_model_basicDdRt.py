# coding: utf8
# !/usr/env/python

import numpy as np
import pytest

from terrainbento import BasicDdRt, NotCoreNodeBaselevelHandler


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
    Kr,
    Kt,
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
        "water_erodibility_lower": Kr,
        "water_erodibility_upper": Kt,
        "water_erosion_rule__threshold": threshold,
        "water_erosion_rule__thresh_depth_derivative": thresh_change_per_depth,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicDdRt(**params)
    for _ in range(100):
        model.run_one_step(1000)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m_sp))) ** (1.0 / n_sp)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m_sp))) ** (1.0 / n_sp)

    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes[22:37])
    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[82:97] > till_predicted_slopes[82:97])
