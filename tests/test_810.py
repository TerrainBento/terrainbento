import os
import numpy as np

from numpy.testing import assert_array_almost_equal

from terrainbento import BasicHyRt
from terrainbento.utilities.utilities import precip_defaults


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0

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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    model = BasicHyRt(params=params)
    for _ in range(2000):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = np.power(
        ((U * v_sc) / (Kr * np.power(actual_areas, m)))
        + (U / (Kr * np.power(actual_areas, m))),
        1. / n,
    )
    till_predicted_slopes = np.power(
        ((U * v_sc) / (Kt * np.power(actual_areas, m)))
        + (U / (Kt * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[22:37], rock_predicted_slopes[22:37])

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[82:97], till_predicted_slopes[82:97])


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 1. / 3.
    n = 2. / 3.
    dt = 10
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0

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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    model = BasicHyRt(params=params)
    for _ in range(2000):
        model.run_one_step(dt)

    # construct actual and predicted slopes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = np.power(
        ((U * v_sc) / (Kr * np.power(actual_areas, m)))
        + (U / (Kr * np.power(actual_areas, m))),
        1. / n,
    )
    till_predicted_slopes = np.power(
        ((U * v_sc) / (Kt * np.power(actual_areas, m)))
        + (U / (Kt * np.power(actual_areas, m))),
        1. / n,
    )

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[22:37], rock_predicted_slopes[22:37])

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[82:97], till_predicted_slopes[82:97])


def test_diffusion_only():

    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.75
    n = 1.0
    dt = 1000
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0

    # construct dictionary. note that D is turned off here
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
        "water_erodability~lower": 0.0,
        "water_erodability~upper": 0.0,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }
    nts = int(total_time / dt)

    reference_node = 9
    # construct and run model
    model = BasicHyRt(params=params)
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
        predicted_z[model.grid.core_nodes], model.z[model.grid.core_nodes]
    )


def test_with_precip_changer():
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    Kr = 0.01
    Kt = 0.001
    v_sc = 0.001
    phi = 0.1
    F_f = 0.0

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
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "settling_velocity": v_sc,
        "sediment_porosity": phi,
        "fraction_fines": F_f,
        "solver": "basic",
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": precip_defaults
    }

    model = BasicHyRt(params=params)
    model._update_erodability_and_threshold_fields()
    assert np.array_equiv(model.eroder.K[model.grid.core_nodes[:8]], Kt) == True
    assert np.array_equiv(model.eroder.K[model.grid.core_nodes[10:]], Kr) == True

    assert "PrecipChanger" in model.boundary_handler
    model.run_one_step(1.0)
    model.run_one_step(1.0)

    true_fw = 10.32628
    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[:8]], Kt * true_fw * np.ones((8))
    )
    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[10:]], Kr * true_fw * np.ones((9))
    )
