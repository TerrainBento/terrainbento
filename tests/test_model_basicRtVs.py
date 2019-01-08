import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicRtVs
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Kss_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 1. / 3.
    n = 2. / 3.
    step = 1000
    initial_soil_thickness = 1.0
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "soil__initial_thickness": initial_soil_thickness,
        "recharge_rate": recharge_rate,
        "hydraulic_conductivity": hydraulic_conductivity,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicRtVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    step = 1000
    initial_soil_thickness = 1.0
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "soil__initial_thickness": initial_soil_thickness,
        "recharge_rate": recharge_rate,
        "hydraulic_conductivity": hydraulic_conductivity,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicRtVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    step = 1000
    initial_soil_thickness = 1.0
    hydraulic_conductivity = 0.1
    recharge_rate = 0.5

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.asc")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 8,
        "number_of_node_columns": 20,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "soil__initial_thickness": initial_soil_thickness,
        "recharge_rate": recharge_rate,
        "hydraulic_conductivity": hydraulic_conductivity,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicRtVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )
