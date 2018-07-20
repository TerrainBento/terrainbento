import os
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal


from terrainbento import BasicRtSa

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_Kss_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 1. / 3.
    n = 2. / 3.
    dt = 1000
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0.0
    soil_transport_decay_depth = 0.5
    soil__initial_thickness = 0.

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
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": soil__initial_thickness,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicRtSa(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[22:37], rock_predicted_slopes[22:37])

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[82:97], till_predicted_slopes[82:97])


def test_steady_Ksp_no_precip_changer():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    dt = 1000
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0.0
    soil_transport_decay_depth = 0.5
    soil__initial_thickness = 0.

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
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": soil__initial_thickness,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicRtSa(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[22:37], rock_predicted_slopes[22:37])

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[82:97], till_predicted_slopes[82:97])


def test_steady_Ksp_no_precip_changer_with_depression_finding():
    U = 0.0001
    Kr = 0.001
    Kt = 0.005
    m = 0.5
    n = 1.0
    dt = 1000
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0.0
    soil_transport_decay_depth = 0.5
    soil__initial_thickness = 0.

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
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": soil__initial_thickness,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
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
    model = BasicRtSa(params=params)
    for _ in range(100):
        model.run_one_step(dt)

    actual_slopes = model.grid.at_node["topographic__steepest_slope"]
    actual_areas = model.grid.at_node["drainage_area"]
    rock_predicted_slopes = (U / (Kr * (actual_areas ** m))) ** (1. / n)
    till_predicted_slopes = (U / (Kt * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[22:37], rock_predicted_slopes[22:37])

    # assert actual and predicted slopes are the same for rock and till portions.
    assert_array_almost_equal(actual_slopes[82:97], till_predicted_slopes[82:97])


def test_diffusion_only():
    U = 0.001
    m = 0.5
    n = 1.0
    dt = 10.
    dx = 10.
    max_soil_production_rate = 0.002
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 1.0
    soil_transport_decay_depth = 0.5
    initial_soil_thickness = 0.0
    number_of_node_rows = 21

    # construct dictionary. note that D is turned off here
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.txt")
    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "dt": dt,
        "output_interval": 2.,
        "run_duration": 200.,
        "number_of_node_rows": number_of_node_rows,
        "number_of_node_columns": 3,
        "node_spacing": dx,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "regolith_transport_parameter": regolith_transport_parameter,
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": initial_soil_thickness,
        "water_erodability~lower": 0,
        "water_erodability~upper": 0,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": m,
        "n_sp": n,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # construct and run model
    model = BasicRtSa(params=params)
    for _ in range(120000):
        model.run_one_step(dt)

    # test steady state soil depth
    actual_depth = model.grid.at_node["soil__depth"][28]
    predicted_depth = -soil_production_decay_depth * np.log(
        U / max_soil_production_rate
    )
    assert_array_almost_equal(actual_depth, predicted_depth, decimal=3)

    # test steady state slope
    actual_profile = model.grid.at_node["topographic__elevation"][model.grid.core_nodes]

    domain = np.arange(0, max(model.grid.node_y + dx), dx)
    steady_domain = np.arange(-max(domain) / 2., max(domain) / 2. + dx, dx)

    half_space = int(len(domain) / 2)
    steady_z_profile_firsthalf = (steady_domain[0:half_space]) ** 2 * U / (
        regolith_transport_parameter
        * 2.
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) - (U * (number_of_node_rows / 2) ** 2) / (
        2.
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )
    steady_z_profile_secondhalf = -(steady_domain[half_space:]) ** 2 * U / (
        regolith_transport_parameter
        * 2.
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    ) + (U * (number_of_node_rows / 2) ** 2) / (
        2.
        * regolith_transport_parameter
        * (1 - np.exp(-predicted_depth / soil_transport_decay_depth))
    )

    steady_z_profile = np.append(
        [-steady_z_profile_firsthalf], [steady_z_profile_secondhalf]
    )
    predicted_profile = steady_z_profile - np.min(steady_z_profile)

    assert_array_almost_equal(actual_profile, predicted_profile[1:-1])


def test_with_precip_changer():
    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.txt")

    Kr = 0.01
    Kt = 0.001
    max_soil_production_rate = 0.0
    soil_production_decay_depth = 0.2
    regolith_transport_parameter = 0.0
    soil_transport_decay_depth = 0.5
    soil__initial_thickness = 0.

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
        "soil_transport_decay_depth": soil_transport_decay_depth,
        "soil_production__maximum_rate": max_soil_production_rate,
        "soil_production__decay_depth": soil_production_decay_depth,
        "soil__initial_thickness": soil__initial_thickness,
        "water_erodability~lower": Kr,
        "water_erodability~upper": Kt,
        "lithology_contact_elevation__file_name": file_name,
        "contact_zone__width": 1.,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "PrecipChanger",
        "PrecipChanger": {
            "daily_rainfall__intermittency_factor": 0.5,
            "daily_rainfall__intermittency_factor_time_rate_of_change": 0.1,
            "daily_rainfall__mean_intensity": 1.0,
            "daily_rainfall__mean_intensity_time_rate_of_change": 0.2,
        },
    }

    model = BasicRtSa(params=params)
    model._update_erodability_field()
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
