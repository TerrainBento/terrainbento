import glob
import os

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from terrainbento import BasicHySt
from terrainbento.utilities import filecmp

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_steady_without_stochastic_duration():
    """Test steady profile solution with fixed duration."""
    U = 0.0001
    K = 0.001
    vs = 1.0e-9
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
        "water_erodability_stochastic": K,
        "v_s": vs,
        "fraction_fines": 1.0,
        "sediment_porosity": 0.0,
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
    model = BasicHySt(params=params)
    for _ in range(100):
        model.run_one_step(step)

    # construct actual and predicted slopes
    ic = model.grid.core_nodes[1:-1]  # "inner" core nodes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"][ic]
    actual_areas = model.grid.at_node["drainage_area"][ic]
    predicted_slopes = (2 * U / (K * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes, predicted_slopes)
