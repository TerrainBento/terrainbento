# coding: utf8
#! /usr/env/python

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal  # assert_array_equal,

from terrainbento import BasicHySa
from terrainbento.utilities import precip_defaults


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    sp_crit_br = 0
    sp_crit_sed = 0
    m = 0.5
    n = 1.0
    dt = 10
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
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~rock": K_rock_sp,
        "water_erodability~sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "H_star": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHySa(params=params)
    for _ in range(800):
        model.run_one_step(dt)

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
    dt = 10
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~rock": K_rock_sp,
        "water_erodability~sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "H_star": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "solver": "basic",
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicHySa(params=params)
    for _ in range(800):
        model.run_one_step(dt)

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


def test_with_precip_changer():
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
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~rock": K_rock_sp,
        "water_erodability~sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "H_star": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults
    }

    model = BasicHySa(params=params)
    assert model.eroder.K_sed[0] == K_sed_sp
    assert "PrecipChanger" in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K_sed, 5) == 0.10326


def test_stability_checker():
    U = 0.0001
    K_rock_sp = 0.001
    K_sed_sp = 0.005
    sp_crit_br = 0
    sp_crit_sed = 0
    m = 0.5
    n = 1.0
    dt = 1000
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
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 3,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "water_erodability~rock": K_rock_sp,
        "water_erodability~sediment": K_sed_sp,
        "sp_crit_br": sp_crit_br,
        "sp_crit_sed": sp_crit_sed,
        "m_sp": m,
        "n_sp": n,
        "v_sc": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "H_star": H_star,
        "solver": "basic",
        "soil__initial_thickness": initial_soil_thickness,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": soil_production__maximum_rate,
        "soil_production__decay_depth": soil_production__decay_depth,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    with pytest.raises(SystemExit):
        model = BasicHySa(params=params)
        for _ in range(800):
            model.run_one_step(dt)


# =============================================================================
# def test_diffusion_only():
#     total_time = 500
#     U = 0.0001
#     K_rock_sp = 0
#     K_sed_sp = 0
#     sp_crit_br = 0
#     sp_crit_sed = 0
#     D = 0.001
#     m = 0.75
#     n = 1.0
#     dt = 1
#     v_sc = 0.001
#     phi = 0.1
#     F_f = 0.1
#     H_star = 0.1
#     initial_soil_thickness = 100
#     soil_transport_decay_depth = 0.1
#     soil_production__maximum_rate = 0
#     soil_production__decay_depth = 0.1
#
#     # construct dictionary. note that D is turned off here
#     params = {'model_grid': 'RasterModelGrid',
#               'dt': dt,
#               'output_interval': 2.,
#               'run_duration': 200.,
#               'number_of_node_rows' : 3,
#               'number_of_node_columns' : 21,
#               'node_spacing' : 100.0,
#               'north_boundary_closed': True,
#               'west_boundary_closed': False,
#               'south_boundary_closed': True,
#               'regolith_transport_parameter': D,
#               'water_erodability~rock': K_rock_sp,
#               'water_erodability~sediment': K_sed_sp,
#               'sp_crit_br': sp_crit_br,
#               'sp_crit_sed': sp_crit_sed,
#               'm_sp': m,
#               'n_sp': n,
#               'v_sc': v_sc,
#               'sediment_porosity': phi,
#               'fraction_fines':F_f,
#               'H_star': H_star,
#               'solver': 'basic',
#               "soil__initial_thickness": initial_soil_thickness,
#               'soil_transport_decay_depth': soil_transport_decay_depth,
#               'soil_production__maximum_rate': soil_production__maximum_rate,
#               'soil_production__decay_depth': soil_production__decay_depth,
#               'random_seed': 3141,
#               'BoundaryHandlers': 'NotCoreNodeBaselevelHandler',
#               'NotCoreNodeBaselevelHandler': {'modify_core_nodes': True,
#                                               'lowering_rate': -U}}
#     nts = int(total_time/dt)
#
#     reference_node = 9
#     # construct and run model
#     model = BasicHySa(params=params)
#     for _ in range(nts):
#         model.run_one_step(dt)
#
#
#     predicted_z = (model.z[model.grid.core_nodes[reference_node]]-(U / (2. * D)) *
#                ((model.grid.x_of_node - model.grid.x_of_node[model.grid.core_nodes[reference_node]])**2))
#
#     # assert actual and predicted elevations are the same.
#     assert_array_almost_equal(predicted_z[model.grid.core_nodes],
#                               model.z[model.grid.core_nodes],
#                               decimal=2)
#
# =============================================================================
