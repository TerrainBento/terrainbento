# coding: utf8
# !/usr/env/python
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicVs, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp,n_sp", [(1.0 / 3, 2.0 / 3.0), (0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
def test_detachment_steady_no_precip_changer(
    clock_simple, grid_1, m_sp, n_sp, depression_finder, U, K
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": 0.001,
        "hydraulic_conductivity": 1.0,
        "depression_finder": depression_finder,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    # construct and run model
    model = BasicVs(**params)
    for _ in range(300):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = (
        U / (params["water_erodibility"] * (actual_areas ** params["m_sp"]))
    ) ** (1.0 / params["n_sp"])

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
    )
