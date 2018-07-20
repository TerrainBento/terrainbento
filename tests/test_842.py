import os
import numpy as np

from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChRtTh

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

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.txt")
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
        "critical_slope": 0.1,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicChRtTh(params=params)
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

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_unit.txt")
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
        "critical_slope": 0.1,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "depression_finder": "DepressionFinderAndRouter",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicChRtTh(params=params)
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
    U = 0.0005
    m = 0.5
    n = 1.0
    dt = 2
    D = 1.0
    S_c = 0.3
    dx = 10.0
    runtime = 30000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.txt")

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "dt": dt,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": dx,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability~lower": 0,
        "water_erodability~upper": 0,
        "water_erosion_rule~upper__threshold": 0.1,
        "water_erosion_rule~lower__threshold": 0.1,
        "contact_zone__width": 1.0,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": S_c,
        "lithology_contact_elevation__file_name": file_name,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # Construct and run model
    model = BasicChRtTh(params=params)
    for _ in range(runtime):
        model.run_one_step(dt)

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
    # print model.grid.at_node['topographic__steepest_slope']
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=3)


def test_with_precip_changer():
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.txt")

    Kr = 0.01
    Kt = 0.001
    Sc = 0.1
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
        "water_erosion_rule~upper__threshold": 0.1,
        "water_erosion_rule~lower__threshold": 0.1,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "critical_slope": Sc,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": {
            "daily_rainfall__intermittency_factor": 0.5,
            "daily_rainfall__intermittency_factor_time_rate_of_change": 0.1,
            "daily_rainfall__mean_intensity": 1.0,
            "daily_rainfall__mean_intensity_time_rate_of_change": 0.2,
        },
    }

    model = BasicChRtTh(params=params)
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
