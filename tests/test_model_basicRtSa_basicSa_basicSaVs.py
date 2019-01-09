import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicSaVs, BasicSa, NotCoreNodeBaselevelHandler, BasicRtSa


_RT_params = {"water_erodability_lower": 0, "water_erodability_upper": 0}
_OTHER_params = {"water_erodability": 0}

@pytest.mark.parametrize("Model,water_params", [ (BasicRtSa, _RT_params), (BasicSa, _OTHER_params), (BasicSaVs, _OTHER_params)])
def test_diffusion_only(clock_simple, grid_4, Model, water_params):

    U = 0.001
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    num_columns = 21

    grid_4.at_node['soil__depth'][:] = 0.

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
    for _ in range(120000):
        model.run_one_step(10)

    dx = grid_4.dx

    # test steady state soil depthf
    actual_depth = model.grid.at_node["soil__depth"][28]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=3)

    # test steady state slope
    actual_profile = model.grid.at_node["topographic__elevation"][21:42]

    domain = np.arange(0, max(model.grid.node_x + dx), dx)
    steady_domain = np.arange(-max(domain) / 2., max(domain) / 2. + dx, dx)

    half_space = int(len(domain) / 2)
    steady_z_profile_firsthalf = (steady_domain[0:half_space]) ** 2 * U / (
        regolith_transport_parameter
        * 2.
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) - (U * (num_columns / 2) ** 2) / (
        2.
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )
    steady_z_profile_secondhalf = -(steady_domain[half_space:]) ** 2 * U / (
        regolith_transport_parameter
        * 2.
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) + (U * (num_columns / 2) ** 2) / (
        2.
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )

    steady_z_profile = np.append(
        [-steady_z_profile_firsthalf], [steady_z_profile_secondhalf]
    )
    predicted_profile = steady_z_profile - np.min(steady_z_profile)

    assert_array_almost_equal(actual_profile, predicted_profile[1:-1])
