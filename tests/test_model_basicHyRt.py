# coding: utf8
# !/usr/env/python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicHyRt, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp,n_sp", [(1.0 / 3, 2.0 / 3.0), (0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("solver", ["basic"])
def test_channel_erosion(
    clock_simple, grid_2, m_sp, n_sp, depression_finder, U, solver
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    phi = 0.1
    F_f = 0.0
    v_sc = 0.001
    Kr = 0.001
    Kt = 0.005
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility_upper": Kt,
        "water_erodibility_lower": Kr,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": solver,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicHyRt(**params)
    for _ in range(4000):
        model.run_one_step(10)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = np.power(
        ((U * v_sc) / (Kr * np.power(actual_areas, m_sp)))
        + (U / (Kr * np.power(actual_areas, m_sp))),
        1.0 / n_sp,
    )
    till_predicted_slopes = np.power(
        ((U * v_sc) / (Kt * np.power(actual_areas, m_sp)))
        + (U / (Kt * np.power(actual_areas, m_sp))),
        1.0 / n_sp,
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37], decimal=3
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97], decimal=3
    )
