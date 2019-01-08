import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChRtTh
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Ksp_no_precip_changer(clock_simple):
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    Tr = 0.0001
    Tt = 0.0005
    m = 0.5
    n = 1.0
    step = 1000

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
        "water_erosion_rule_upper__threshold": Tt,
        "water_erosion_rule_lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "critical_slope": 0.1,
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
    model = BasicChRtTh(params=params)
    for _ in range(200):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual
    # slopes to fall.
    rock_predicted_slopes_upper = ((U + Tr) / (Kr * (actual_areas ** m))) ** (
        1. / n
    )
    till_predicted_slopes_upper = ((U + Tt) / (Kt * (actual_areas ** m))) ** (
        1. / n
    )

    rock_predicted_slopes_lower = ((U + 0.) / (Kr * (actual_areas ** m))) ** (
        1. / n
    )
    till_predicted_slopes_lower = ((U + 0.) / (Kt * (actual_areas ** m))) ** (
        1. / n
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes_lower[22:37])

    assert np.all(actual_slopes[22:37] < rock_predicted_slopes_upper[22:37])

    assert np.all(actual_slopes[82:97] > till_predicted_slopes_lower[82:97])

    assert np.all(actual_slopes[82:97] < till_predicted_slopes_upper[82:97])


def test_steady_Ksp_no_precip_changer_with_depression_finding(clock_simple):
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    Tr = 0.001
    Tt = 0.005
    m = 0.5
    n = 1.0
    step = 1000

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
        "water_erosion_rule_upper__threshold": Tt,
        "water_erosion_rule_lower__threshold": Tr,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "critical_slope": 0.1,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "depression_finder": "DepressionFinderAndRouter",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicChRtTh(params=params)
    for _ in range(200):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]

    # note that since we have a smooth threshold, we do not have a true
    # analytical solution, but a bracket within wich we expect the actual
    # slopes to fall.
    rock_predicted_slopes_upper = ((U + Tr) / (Kr * (actual_areas ** m))) ** (
        1. / n
    )
    till_predicted_slopes_upper = ((U + Tt) / (Kt * (actual_areas ** m))) ** (
        1. / n
    )

    rock_predicted_slopes_lower = ((U + 0.) / (Kr * (actual_areas ** m))) ** (
        1. / n
    )
    till_predicted_slopes_lower = ((U + 0.) / (Kt * (actual_areas ** m))) ** (
        1. / n
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert np.all(actual_slopes[22:37] > rock_predicted_slopes_lower[22:37])
    assert np.all(actual_slopes[22:37] < rock_predicted_slopes_upper[22:37])
    assert np.all(actual_slopes[82:97] > till_predicted_slopes_lower[82:97])
    assert np.all(actual_slopes[82:97] < till_predicted_slopes_upper[82:97])
