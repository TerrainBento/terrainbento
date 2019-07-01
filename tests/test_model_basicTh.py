import numpy as np
import pytest

from terrainbento import BasicTh, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("m_sp,n_sp", [(0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
def test_steady_Kss_no_precip_changer(
    clock_simple, grid_2, U, K, m_sp, n_sp, depression_finder
):

    threshold = 0.01

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )

    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "water_erodibility": K,
        "water_erosion_rule__threshold": threshold,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicTh(**params)
    for _ in range(200):
        model.run_one_step(1000)

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
