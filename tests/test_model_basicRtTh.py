# coding: utf8
#! /usr/env/python

import os
import numpy as np

from numpy.testing import assert_array_almost_equal

from terrainbento import BasicRtTh
from terrainbento.utilities import precip_defaults, precip_testing_factor


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    Tr = 0.0001
    Tt = 0.0005
    m = 0.5
    n = 1.0
    dt = 1000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule~upper__threshold": Tt,
        "water_erosion_rule~lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicRtTh(params=params)
    for _ in range(200):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual slopes
    # to fall.
    rock_predicted_slopes_upper = ((U + Tr) / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes_upper = ((U + Tt) / (Kt * (actual_areas ** m))) ** (1. / n)

    rock_predicted_slopes_lower = ((U + 0.) / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes_lower = ((U + 0.) / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till portions.
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes_lower[22:37]) == True
    assert np.all(actual_slopes[22:37] < rock_predicted_slopes_upper[22:37]) == True

    assert np.all(actual_slopes[82:97] > till_predicted_slopes_lower[82:97]) == True
    assert np.all(actual_slopes[82:97] < till_predicted_slopes_upper[82:97]) == True


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    Tr = 0.001
    Tt = 0.005
    m = 0.5
    n = 1.0
    dt = 1000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule~upper__threshold": Tt,
        "water_erosion_rule~lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "depression_finder": "DepressionFinderAndRouter",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicRtTh(params=params)
    for _ in range(200):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual slopes
    # to fall.
    rock_predicted_slopes_upper = ((U + Tr) / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes_upper = ((U + Tt) / (Kt * (actual_areas ** m))) ** (1. / n)

    rock_predicted_slopes_lower = ((U + 0.) / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes_lower = ((U + 0.) / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till portions.
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes_lower[22:37]) == True
    assert np.all(actual_slopes[22:37] < rock_predicted_slopes_upper[22:37]) == True

    assert np.all(actual_slopes[82:97] > till_predicted_slopes_lower[82:97]) == True
    assert np.all(actual_slopes[82:97] < till_predicted_slopes_upper[82:97]) == True


def test_diffusion_only():
    total_time = 5.0e6
    U = 0.0001
    Kr = 0.
    Kt = 0.
    Tr = 0.000001
    Tt = 0.000001
    m = 0.5
    n = 1.0
    dt = 1000
    D = 1

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule~upper__threshold": Tt,
        "water_erosion_rule~lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }
    nts = int(total_time / dt)

    reference_node = 9
    # construct and run model
    model = BasicRtTh(params=params)
    for _ in range(nts):
        model.run_one_step(dt)

    predicted_z = model.z[model.grid.core_nodes[reference_node]] - (U / (2. * D)) * (
        (
            model.grid.x_of_node
            - model.grid.x_of_node[model.grid.core_nodes[reference_node]]
        )
        ** 2
    )

    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(
        predicted_z[model.grid.core_nodes], model.z[model.grid.core_nodes], decimal=2
    )


def test_with_precip_changer():
    Kr = 0.001
    Kt = 0.005
    Tr = 0.01
    Tt = 0.05

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": 1,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule~upper__threshold": Tt,
        "water_erosion_rule~lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults,
    }

    model = BasicRtTh(params=params)
    model._update_erodability_and_threshold_fields()
    assert np.array_equiv(model.eroder.K[model.grid.core_nodes[:8]], Kt) == True
    assert np.array_equiv(model.eroder.K[model.grid.core_nodes[10:]], Kr) == True

    assert "PrecipChanger" in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)

    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[:8]],
        Kt * precip_testing_factor * np.ones((8)),
    )
    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[10:]],
        Kr * precip_testing_factor * np.ones((9)),
    )
