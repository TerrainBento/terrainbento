import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    BasicRtSa,
    BasicSa,
    BasicSaVs,
    NotCoreNodeBaselevelHandler,
)

_RT_params = {"water_erodibility_lower": 0, "water_erodibility_upper": 0}
_OTHER_params = {"water_erodibility": 0}


@pytest.mark.parametrize(
    "Model,water_params",
    [
        (BasicSa, _OTHER_params),
        (BasicRtSa, _RT_params),
        (BasicSaVs, _OTHER_params),
    ],
)
def test_diffusion_only(clock_simple, grid_4, Model, water_params):

    U = 0.001
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5

    grid_4.at_node["soil__depth"][:] = 0.0

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_4, modify_core_nodes=True, lowering_rate=-U
    )

    params = {
        "grid": grid_4,
        "clock": clock_simple,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }
    for p in water_params:
        params[p] = water_params[p]

    # construct and run model
    model = Model(**params)
    for _ in range(20000):
        model.run_one_step(15)

    dx = grid_4.dx

    # test steady state soil depthf
    actual_depth = model.grid.at_node["soil__depth"][28]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=2)

    # test steady state slope
    actual_profile = model.grid.at_node["topographic__elevation"][21:42]

    domain = np.arange(0, max(model.grid.node_x + dx), dx)

    half_domain = np.arange(0, max(domain) / 2.0 + dx, dx)

    one_minus_h_hstar = 1 - np.exp(
        -predicted_depth / soil_transport_decay_depth
    )

    half_domain_z = (
        -half_domain ** 2
        * U
        / (
            regolith_transport_parameter
            * soil_transport_decay_depth
            * 2.0
            * one_minus_h_hstar
        )
    )

    steady_z_profile = np.concatenate(
        (np.flipud(half_domain_z), half_domain_z[1:])
    )

    predicted_profile = steady_z_profile - np.min(steady_z_profile)

    assert_array_almost_equal(actual_profile, predicted_profile, decimal=1)
