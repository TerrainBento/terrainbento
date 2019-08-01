# coding: utf8
# !/usr/env/python

# coding: utf8
# !/usr/env/python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicHySa, NotCoreNodeBaselevelHandler, PrecipChanger


@pytest.mark.parametrize("m_sp,n_sp", [(1.0 / 3, 2.0 / 3.0), (0.5, 1.0)])
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
@pytest.mark.parametrize("solver", ["basic"])
def test_channel_erosion(
    clock_simple, grid_1, m_sp, n_sp, depression_finder, U, solver
):
    ncnblh = NotCoreNodeBaselevelHandler(
        grid_1, modify_core_nodes=True, lowering_rate=-U
    )
    phi = 0.1
    F_f = 0.0
    v_sc = 0.001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    sp_crit_br = 0
    sp_crit_sed = 0
    H_star = 0.1
    soil_transport_decay_depth = 1
    soil_production__maximum_rate = 0
    soil_production__decay_depth = 0.5
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility_rock": K_rock_sp,
        "water_erodibility_sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m_sp,
        "n_sp": n_sp,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "roughness__length_scale": H_star,
        "solver": solver,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicHySa(**params)
    for _ in range(2000):
        model.run_one_step(10)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["surface_water__discharge"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K_sed_sp * np.power(actual_areas, m_sp)))
        + (U / (K_rock_sp * np.power(actual_areas, m_sp))),
        1.0 / n_sp,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )

    with pytest.raises(SystemExit):
        for _ in range(800):
            model.run_one_step(100000)


def test_with_precip_changer(
    clock_simple, grid_1, precip_defaults, precip_testing_factor
):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility_rock": 0.001,
        "water_erodibility_sediment": 0.01,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }
    model = BasicHySa(**params)

    assert model.eroder.K_sed[0] == params["water_erodibility_sediment"]
    assert model.eroder.K_br[0] == params["water_erodibility_rock"]
    assert "PrecipChanger" in model.boundary_handlers
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K_sed, 5) == round(
        params["water_erodibility_sediment"] * precip_testing_factor, 5
    )
    assert round(model.eroder.K_br, 5) == round(
        params["water_erodibility_rock"] * precip_testing_factor, 5
    )
