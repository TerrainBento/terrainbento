import glob
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_equal

from terrainbento import BasicStVs

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_bad_transmiss():
    """Test steady profile solution with fixed duration."""
    K = 0.001
    H0 = 0.0
    Ks = 0.0
    m = 1.0
    n = 1.0

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
        "m_sp": m,
        "n_sp": n,
        "soil__initial_thickness": H0,
        "hydraulic_conductivity": Ks,
        "number_of_sub_time_steps": 100,
        "rainfall_intermittency_factor": 1.0,
        "rainfall__mean_rate": 1.0,
        "rainfall__shape_factor": 1.0,
    }

    with pytest.raises(ValueError):
        BasicStVs(params=params)


def test_steady_without_stochastic_duration():
    """Test steady profile solution with fixed duration."""
    U = 0.0001
    K = 0.001
    H0 = 1.0e-9
    Ks = 1.0e-9
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
        "m_sp": m,
        "n_sp": n,
        "soil__initial_thickness": H0,
        "hydraulic_conductivity": Ks,
        "number_of_sub_time_steps": 100,
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
    model = BasicStVs(params=params)
    for _ in range(100):
        model.run_one_step(step)

    # construct actual and predicted slopes
    ic = model.grid.core_nodes[1:-1]  # "inner" core nodes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"][ic]
    actual_areas = model.grid.at_node["drainage_area"][ic]
    predicted_slopes = (U / (K * (actual_areas ** m))) ** (1. / n)

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes, predicted_slopes)
