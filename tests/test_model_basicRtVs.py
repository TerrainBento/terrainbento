import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicRtVs, NotCoreNodeBaselevelHandler
from terrainbento.utilities import filecmp


@pytest.mark.parametrize("m_sp", [1. / 3, 0.5, 0.75, 0.25])
@pytest.mark.parametrize("n_sp", [2. / 3., 1.])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"])
def test_steady_Kss_no_precip_changer(clock_simple, grid_2, U, Kr, Kt, m_sp, n_sp, depression_finder):

    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )

    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "recharge_rate": recharge_rate,
        "hydraulic_conductivity": hydraulic_conductivity,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicRtVs(**params)
    for _ in range(100):
        model.run_one_step(1000)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m_sp))) ** (1. / n_sp)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m_sp))) ** (1. / n_sp)

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )
