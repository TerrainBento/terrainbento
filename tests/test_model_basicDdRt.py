import os
import numpy as np

from numpy.testing import assert_array_almost_equal

from terrainbento import BasicDdRt
from terrainbento.utilities import *


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# def test_steady_Ksp_no_precip_changer_no_thresh():
#     U = 0.0001
#     Kr = 0.001
#     Kt = 0.005
#     T = 0.0
#     dTdz = 0.0
#     m = 0.5
#     n = 1.0
#     dt = 1000
#
#     file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
#     # construct dictionary. note that D is turned off here
#     params = {
#         "model_grid": "RasterModelGrid",
#         "clock": {"dt": 1,
#         "output_interval": 2.,
#         "run_duration": 200.},
#         "number_of_node_rows": 8,
#         "number_of_node_columns": 20,
#         "node_spacing": 100.0,
#         "north_boundary_closed": True,
#         "south_boundary_closed": True,
#         "regolith_transport_parameter": 0.,
#         "water_erodability~lower": Kr,
#         "water_erodability~upper": Kt,
#         "water_erosion_rule__threshold": T,
#         "water_erosion_rule__thresh_depth_derivative": dTdz,
#         "lithology_contact_elevation__file_name": file_name,
#         "contact_zone__width": 1.,
#         "m_sp": m,
#         "n_sp": n,
#         "random_seed": 3141,
#         "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
#         "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
#     }
#
#     # construct and run model
#     model = BasicDdRt(params=params)
#     for _ in range(100):
#         model.run_one_step(dt)
#
#     actual_slopes = model.grid.at_node["topographic__steepest_slope"]
#     actual_areas = model.grid.at_node["drainage_area"]
#     rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
#     till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)
#
#     # assert slopes are correct
#     assert_array_almost_equal(
#         actual_slopes[model.grid.core_nodes[22:37]],
#         predicted_slopes[model.grid.core_nodes[22:37]],
#     )
#
#     assert_array_almost_equal(
#         actual_slopes[model.grid.core_nodes[82:97]],
#         predicted_slopes[model.grid.core_nodes[82:97]],
#     )


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    T = 0.001
    dTdz = 0.005
    m = 0.5
    n = 1.0
    dt = 1000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule__threshold": T,
        "water_erosion_rule__thresh_depth_derivative": dTdz,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicDdRt(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes[22:37]) == True

    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[82:97] > till_predicted_slopes[82:97]) == True


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    T = 0.001
    dTdz = 0.005
    m = 0.5
    n = 1.0
    dt = 1000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule__threshold": T,
        "water_erosion_rule__thresh_depth_derivative": dTdz,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicDdRt(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes[22:37]) == True

    # assert actual slopes are steeper than simple stream power prediction
    assert np.all(actual_slopes[82:97] > till_predicted_slopes[82:97]) == True


def test_diffusion_only():
    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.5
    n = 1.0
    dt = 1000
    T = 0.001
    dTdz = 0.005

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability~lower": 0,
        "water_erodability~upper": 0,
        "water_erosion_rule__threshold": T,
        "water_erosion_rule__thresh_depth_derivative": dTdz,
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
    model = BasicDdRt(params=params)
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
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    Kr = 0.01
    Kt = 0.001

    T = 0.001
    dTdz = 0.005

    params = {
        "model_grid": "RasterModelGrid",
        "clock": SIMPLE_CLOCK,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "water_erosion_rule__threshold": T,
        "water_erosion_rule__thresh_depth_derivative": dTdz,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults,
    }

    model = BasicDdRt(params=params)
    model._update_erodability_field()
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
