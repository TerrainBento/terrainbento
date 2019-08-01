import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    Basic,
    BasicCh,
    BasicChRt,
    BasicChRtTh,
    BasicChSa,
    BasicCv,
    BasicRt,
    BasicRtSa,
    BasicRtTh,
    BasicSa,
    NotCoreNodeBaselevelHandler,
)


@pytest.mark.parametrize("Model", [BasicRt, BasicChRt, BasicRtSa])
@pytest.mark.parametrize("m_sp", [1.0 / 3, 0.5, 0.75, 0.25])
@pytest.mark.parametrize("n_sp", [2.0 / 3.0, 1.0])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize(
    "flow_director", ["FlowDirectorSteepest", "FlowDirectorD8"]
)
def test_rock_till_steady_no_precip_changer(
    clock_simple,
    grid_2,
    m_sp,
    n_sp,
    depression_finder,
    U,
    Kr,
    Kt,
    Model,
    flow_director,
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
        "depression_finder": depression_finder,
        "flow_director": flow_director,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = Model(**params)
    for _ in range(200):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m_sp))) ** (1.0 / n_sp)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m_sp))) ** (1.0 / n_sp)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37], decimal=4
    )

    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97], decimal=4
    )


@pytest.mark.parametrize("Model", [BasicRtTh, BasicChRtTh])
@pytest.mark.parametrize("m_sp", [1.0 / 3, 0.5, 0.75, 0.25])
@pytest.mark.parametrize("n_sp", [1.0])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize(
    "flow_director", ["FlowDirectorSteepest", "FlowDirectorD8"]
)
def test_rock_till_steady_no_precip_changer_ChRtTh(
    clock_simple,
    Model,
    grid_2,
    m_sp,
    n_sp,
    depression_finder,
    U,
    Kr,
    Kt,
    flow_director,
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
        "depression_finder": depression_finder,
        "flow_director": flow_director,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "water_erosion_rule_upper__threshold": 0.0000001,
        "water_erosion_rule_lower__threshold": 0.0000001,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = Model(**params)
    for _ in range(200):
        model.run_one_step(1000)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m_sp))) ** (1.0 / n_sp)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m_sp))) ** (1.0 / n_sp)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37], decimal=4
    )

    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97], decimal=4
    )


@pytest.mark.parametrize(
    "Model", [Basic, BasicCv, BasicCh, BasicChSa, BasicSa]
)
@pytest.mark.parametrize("m_sp", [1.0 / 3, 0.5, 0.75, 0.25])
@pytest.mark.parametrize("n_sp", [2.0 / 3.0, 1.0])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize(
    "flow_director", ["FlowDirectorSteepest", "FlowDirectorD8"]
)
def test_detachment_steady_no_precip_changer(
    clock_simple,
    grid_1,
    m_sp,
    n_sp,
    depression_finder,
    U,
    K,
    Model,
    flow_director,
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": 0.001,
        "depression_finder": depression_finder,
        "flow_director": flow_director,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    # construct and run model
    model = Model(**params)
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
