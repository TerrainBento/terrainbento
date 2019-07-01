import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicStVs, NotCoreNodeBaselevelHandler


def test_bad_transmiss(grid_2, clock_simple):
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "hydraulic_conductivity": 0.0,
    }

    with pytest.raises(ValueError):
        BasicStVs(**params)


@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("m_sp,n_sp", [(1, 1)])
def test_steady_without_stochastic_duration(
    clock_simple, depression_finder, U, K, grid_2, m_sp, n_sp
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_2, modify_core_nodes=True, lowering_rate=-U
    )
    grid_2.at_node["soil__depth"][:] = 1.0e-9
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "hydraulic_conductivity": 1.0e-9,
        "number_of_sub_time_steps": 100,
        "rainfall_intermittency_factor": 1.0,
        "rainfall__mean_rate": 1.0,
        "rainfall__shape_factor": 1.0,
        "random_seed": 3141,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicStVs(**params)
    for _ in range(100):
        model.run_one_step(1.0)

    # construct actual and predicted slopes
    ic = model.grid.core_nodes[1:-1]  # "inner" core nodes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = (U / (K * (actual_areas ** m_sp))) ** (1.0 / n_sp)
    assert_array_almost_equal(
        actual_slopes[ic], predicted_slopes[ic], decimal=4
    )
