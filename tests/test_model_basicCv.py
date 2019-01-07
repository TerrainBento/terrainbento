# coding: utf8
# !/usr/env/python
import pytest

from numpy.testing import assert_array_almost_equal

from terrainbento import BasicCv, NotCoreNodeBaselevelHandler, PrecipChanger


@pytest.mark.parametrize("m_sp", [1. / 3, 0.5, 0.75, 0.25])
@pytest.mark.parametrize("n_sp", [2. / 3., 1.])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
def test_basic_steady_no_precip_changer(
    clock_simple, grid_1, m_sp, n_sp, depression_finder, U, K
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": 0.001,
        "depression_finder": depression_finder,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "climate_factor": 0.5,
        "climate_constant_date": 10,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    # construct and run model
    model = BasicCv(**params)
    for _ in range(200):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = (
        U / (params["water_erodability"] * (actual_areas ** params["m_sp"]))
    ) ** (1. / params["n_sp"])

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
    )


def test_diffusion_only(clock_simple, grid_1, U):
    total_time = 5.0e6
    step = 1000
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 1,
        "water_erodability": 0,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    # construct and run model
    model = BasicCv(**params)

    nts = int(total_time / step)
    for _ in range(nts):
        model.run_one_step(1000)
    reference_node = 9
    predicted_z = model.z[model.grid.core_nodes[reference_node]] - (
        U / (2. * params["regolith_transport_parameter"])
    ) * (
        (
            model.grid.x_of_node
            - model.grid.x_of_node[model.grid.core_nodes[reference_node]]
        )
        ** 2
    )

    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(
        predicted_z[model.grid.core_nodes],
        model.z[model.grid.core_nodes],
        decimal=2,
    )
