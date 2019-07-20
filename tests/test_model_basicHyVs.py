# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicHyVs, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp,n_sp", [(1.0 / 3, 2.0 / 3.0), (0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("solver", ["basic", "adaptive"])
def test_channel_erosion(
    clock_simple, grid_1, m_sp, n_sp, depression_finder, U, K, solver
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    phi = 0.0
    F_f = 0.0
    v_sc = 0.001
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "hydraulic_conductivity": 0.0,
        "solver": solver,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicHyVs(**params)
    for _ in range(3000):
        model.run_one_step(5)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K * np.power(actual_areas, m_sp)))
        + (U / (K * np.power(actual_areas, m_sp))),
        1.0 / n_sp,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )
