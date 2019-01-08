# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicHySa, PrecipChanger
from terrainbento.utilities import filecmp


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    sp_crit_br = 0
    sp_crit_sed = 0
    m = 0.5
    n = 1.0
    step = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
    H_star = 0.1
    initial_soil_thickness = 0
    soil_transport_decay_depth = 1
    soil_production__maximum_rate = 0
    soil_production__decay_depth = 0.5
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_rock": K_rock_sp,
        "water_erodability_sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "roughness__length_scale": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHySa(params=params)
    for _ in range(800):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K_sed_sp * np.power(actual_areas, m)))
        + (U / (K_rock_sp * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.01
    sp_crit_br = 0
    sp_crit_sed = 0
    m = 0.5
    n = 1.0
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
    H_star = 0.1
    initial_soil_thickness = 0
    soil_transport_decay_depth = 1
    soil_production__maximum_rate = 0
    soil_production__decay_depth = 0.5
    step = 10
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_rock": K_rock_sp,
        "water_erodability_sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "roughness__length_scale": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicHySa(params=params)
    for _ in range(800):
        model.run_one_step(step)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    predicted_slopes = np.power(
        ((U * v_sc) / (K_sed_sp * np.power(actual_areas, m)))
        + (U / (K_rock_sp * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(
        actual_slopes[model.grid.core_nodes[1:-1]],
        predicted_slopes[model.grid.core_nodes[1:-1]],
        decimal=4,
    )


def test_with_precip_changer(
    clock_simple, grid_1, precip_defaults, precip_testing_factor
):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability_rock": 0.001,
        "water_erodability_sediment": 0.01,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }
    model = BasicHySa(**params)

    assert model.eroder.K_sed[0] == params["water_erodability_sediment"]
    assert model.eroder.K_br[0] == params["water_erodability_rock"]
    assert "PrecipChanger" in model.boundary_handlers
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K_sed, 5) == round(
        params["water_erodability_sediment"] * precip_testing_factor, 5
    )
    assert round(model.eroder.K_br, 5) == round(
        params["water_erodability_rock"] * precip_testing_factor, 5
    )


def test_stability_checker():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    sp_crit_br = 0
    sp_crit_sed = 0
    m = 0.5
    n = 1.0
    step = 1000
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0
    H_star = 0.1
    regolith_transport_parameter = 1
    initial_soil_thickness = 1
    soil_transport_decay_depth = 1
    soil_production__maximum_rate = 0.001
    soil_production__decay_depth = 0.5
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "water_erodability_rock": K_rock_sp,
        "water_erodability_sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "roughness__length_scale": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    with pytest.raises(SystemExit):
        model = BasicHySa(params=params)
        for _ in range(800):
            model.run_one_step(step)
