import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChRt
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Kss_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 1. / 3.
    n = 2. / 3.
    Sc = 0.1
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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": Sc,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicChRt(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )


def test_steady_Ksp_no_precip_changer(clock_simple):
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    Sc = 0.1
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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": Sc,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicChRt(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )


def test_steady_Ksp_no_precip_changer_with_depression_finding(clock_simple):
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    step = 1000
    Sc = 0.1

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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "critical_slope": Sc,
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
    model = BasicChRt(params=params)
    for _ in range(100):
        model.run_one_step(step)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[22:37], rock_predicted_slopes[22:37]
    )

    # assert actual and predicted slopes are the same for rock and till
    # portions.
    assert_array_almost_equal(
        actual_slopes[82:97], till_predicted_slopes[82:97]
    )


def test_diffusion_only():
    U = 0.0005
    m = 0.5
    n = 1.0
    step = 2
    D = 1.0
    S_c = 0.3
    dx = 10.0
    runtime = 30000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"step": step, "output_interval": 2., "stop": 200.},
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": dx,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability_lower": 0,
        "water_erodability_upper": 0,
        "contact_zone__width": 1.0,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": S_c,
        "lithology_contact_elevation__file_name": file_name,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # Construct and run model
    model = BasicChRt(params=params)
    for _ in range(runtime):
        model.run_one_step(step)

    # Construct actual and predicted slope at top edge of domain
    x = 8.5 * dx
    qs = U * x
    nterms = 11
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = D * (1 / (S_c ** (2 * (k - 1))))
    p = np.fliplr([p])[0]
    p = np.append(p, qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    # print(predicted_slope)

    actual_slope = np.abs(model.grid.at_node["topographic__steepest_slope"][7])
    # print model.grid.at_node["topographic__steepest_slope"]
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=3)


def test_with_precip_changer(
    clock_simple, precip_defaults, precip_testing_factor
):
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    Kr = 0.01
    Kt = 0.001
    Sc = 0.1
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "critical_slope": Sc,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults,
    }

    model = BasicChRt(params=params)
    model._update_erodability_field()
    assert (
        np.array_equiv(model.eroder.K[model.grid.core_nodes[:8]], Kt) is True
    )
    assert (
        np.array_equiv(model.eroder.K[model.grid.core_nodes[10:]], Kr) is True
    )

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
