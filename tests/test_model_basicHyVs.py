# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicHyVs, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp", [1. / 3, 0.5])
@pytest.mark.parametrize("n_sp", [2. / 3., 1.])
@pytest.mark.parametrize("depression_finder", [None, "DepressionFinderAndRouter"])
@pytest.mark.parametrize("solver", ["basic", "adaptive"])
def test_no_precip_changer(
    clock_simple, grid_2, m_sp, n_sp, depression_finder, U, K, solver
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    phi = 0.1
    F_f = 0.0
    v_sc = 0.001
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "hydraulic_conductivity": 0.,
        "solver": solver,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicHyVs(**params)
    for _ in range(2000):
        model.run_one_step(10)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m_sp)))
        + (U / (K * np.power(actual_areas, m_sp))),
        1. / n_sp,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )
