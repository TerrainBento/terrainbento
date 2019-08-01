# coding: utf8
# !/usr/env/python

import numpy as np
import pytest

from terrainbento import BasicRtTh, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp,n_sp", [(0.75, 1.0), (0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
def test_steady_Ksp_no_precip_changer(
    clock_simple, grid_2, depression_finder, m_sp, n_sp
):
    U = 0.0001
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )

    Kr = 0.001
    Kt = 0.005
    Tr = 0.0001
    Tt = 0.0005

    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility_lower": Kr,
        "water_erodibility_upper": Kt,
        "water_erosion_rule_upper__threshold": Tt,
        "water_erosion_rule_lower__threshold": Tr,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicRtTh(**params)
    for _ in range(200):
        model.run_one_step(1000)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]

    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual
    # slopes to fall.
    rock_predicted_slopes_upper = (
        (U + Tr) / (Kr * (actual_areas ** m_sp))
    ) ** (1.0 / n_sp)
    till_predicted_slopes_upper = (
        (U + Tt) / (Kt * (actual_areas ** m_sp))
    ) ** (1.0 / n_sp)

    rock_predicted_slopes_lower = (
        (U + 0.0) / (Kr * (actual_areas ** m_sp))
    ) ** (1.0 / n_sp)
    till_predicted_slopes_lower = (
        (U + 0.0) / (Kt * (actual_areas ** m_sp))
    ) ** (1.0 / n_sp)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes_lower[22:37])
    assert np.all(actual_slopes[22:37] < rock_predicted_slopes_upper[22:37])

    assert np.all(actual_slopes[82:97] > till_predicted_slopes_lower[82:97])

    assert np.all(actual_slopes[82:97] < till_predicted_slopes_upper[82:97])
