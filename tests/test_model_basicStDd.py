import glob
import os

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from terrainbento import BasicDdSt

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_without_stochastic_duration():
    """Test steady profile solution with fixed duration."""
    U = 0.0001
    K = 0.001
    thresh = 1.0e-9
    m = 1.0
    n = 1.0
    step = 1.0

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 6,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~stochastic": K,
        "water_erosion_rule__threshold": thresh,
        "thresh_change_per_depth": 0.0,
        "m_sp": m,
        "n_sp": n,
        "number_of_sub_time_steps": 100,
        "infiltration_capacity": 1.0,
        "rainfall_intermittency_factor": 1.0,
        "rainfall__mean_rate": 1.0,
        "rainfall__shape_factor": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicDdSt(params=params)
    for _ in range(100):
        model.run_one_step(step)

    # construct actual and predicted slopes
    ic = model.grid.core_nodes[1:-1]  # "inner" core nodes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"][ic]
    actual_areas = model.grid.at_node["drainage_area"][ic]
    predicted_slopes = (2 * U / (K * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes, predicted_slopes)


def test_stochastic_duration_rainfall_means():
    """Test option with stochastic duration.

    Test is simply to get the correct total cumulative rain depth.
    """
    U = 0.0001
    K = 0.0001
    thresh = 0.001
    m = 1.0
    n = 1.0
    step = 200.0

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"step": step, "output_interval": 401., "stop": 400.},
        "number_of_node_rows": 3,
        "number_of_node_columns": 6,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "south_boundary_closed": True,
        "regolith_transport_parameter": 0.,
        "water_erodability~stochastic": K,
        "water_erosion_rule__threshold": thresh,
        "thresh_change_per_depth": 0.0,
        "m_sp": m,
        "n_sp": n,
        "opt_stochastic_duration": True,
        "record_rain": True,
        "mean_storm_duration": 1.0,
        "mean_interstorm_duration": 1.0,
        "infiltration_capacity": 1.0,
        "random_seed": 3141,
        "mean_storm_depth": 1.0,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }

    # construct and run model
    model = BasicDdSt(params=params)
    model.run()

    cum_rain_depth = np.sum(
        np.array(model.rain_record["event_duration"])
        * np.array(model.rain_record["rainfall_rate"])
    )
    assert_equal(np.round(cum_rain_depth), 200.0)

    os.remove("storm_sequence.txt")
    fs = glob.glob(model._out_file_name + "*.nc")
    for f in fs:
        os.remove(f)


def test_diffusion_only():
    total_time = 5.0e6
    U = 0.001
    D = 1
    m = 0.75
    n = 1.0
    step = 1000

    # construct dictionary. note that D is turned off here
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "number_of_node_rows": 3,
        "number_of_node_columns": 21,
        "node_spacing": 100.0,
        "north_boundary_closed": True,
        "west_boundary_closed": False,
        "south_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability~stochastic": 0.0,
        "water_erosion_rule__threshold": 1.0e-9,
        "thresh_change_per_depth": 0.0,
        "m_sp": m,
        "n_sp": n,
        "number_of_sub_time_steps": 100,
        "infiltration_capacity": 1.0,
        "rainfall_intermittency_factor": 1.0,
        "rainfall__mean_rate": 1.0,
        "rainfall__shape_factor": 1.0,
        "random_seed": 3141,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {
            "modify_core_nodes": True,
            "lowering_rate": -U,
        },
    }
    nts = int(total_time / step)

    reference_node = 9
    # construct and run model
    model = BasicDdSt(params=params)
    for _ in range(nts):
        model.run_one_step(step)

    predicted_z = model.z[model.grid.core_nodes[reference_node]] - (
        U / (2. * D)
    ) * (
        (
            model.grid.x_of_node
            - model.grid.x_of_node[model.grid.core_nodes[reference_node]]
        )
        ** 2
    )

    # assert actual and predicted elevations are the same.
    assert_array_almost_equal(
        predicted_z[model.grid.core_nodes],
        model.z[model.grid.core_nodes],
        decimal=2,
    )
