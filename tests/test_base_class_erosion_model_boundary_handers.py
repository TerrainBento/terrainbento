# coding: utf8
# !/usr/env/python

import numpy as np
import pytest
from numpy.testing import assert_array_equal  # , assert_array_almost_equal

from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from terrainbento import Basic, BasicSt, ErosionModel
from terrainbento.boundary_handlers import (
    CaptureNodeBaselevelHandler,
    GenericFuncBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    SingleNodeBaselevelHandler,
)


@pytest.mark.parametrize("keyword", ["BasicSt", "NotCoreNodeBaselevelHandler"])
def test_bad_boundary_condition_string(
    clock_simple, almost_default_grid, keyword
):
    params = {
        "grid": almost_default_grid,
        "clock": clock_simple,
        "boundary_handlers": {keyword: BasicSt},
    }
    with pytest.raises(ValueError):
        ErosionModel(**params)


def test_bad_boundary_condition_yaml(bad_handler_yaml, tmpdir):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(bad_handler_yaml)

        with pytest.raises(ValueError):
            ErosionModel.from_file("./params.yaml")


def test_single_node_blh_with_closed_boundaries(
    clock_simple, simple_square_grid
):
    snblh = SingleNodeBaselevelHandler(
        simple_square_grid,
        modify_outlet_node=False,
        lowering_rate=-0.0005,
        outlet_id=3,
    )

    params = {
        "clock": clock_simple,
        "grid": simple_square_grid,
        "boundary_handlers": {"SingleNodeBaselevelHandler": snblh},
    }
    model = Basic(**params)
    assert model.grid.status_at_node[3] == FIXED_VALUE_BOUNDARY


def test_pass_two_boundary_handlers(clock_simple, simple_square_grid, U):
    ncnblh = NotCoreNodeBaselevelHandler(
        simple_square_grid, modify_core_nodes=True, lowering_rate=-U
    )
    snblh = SingleNodeBaselevelHandler(
        simple_square_grid, modify_outlet_node=False, lowering_rate=-U
    )
    params = {
        "grid": simple_square_grid,
        "clock": clock_simple,
        "boundary_handlers": {"mybh1": ncnblh, "mybh2": snblh},
    }
    model = Basic(**params)
    model.run_one_step(1.0)

    truth = np.zeros(model.z.size)
    truth[0] -= U
    truth[model.grid.core_nodes] += U
    assert_array_equal(model.z, truth)

    status_at_node = np.zeros(model.z.size)
    status_at_node[model.grid.boundary_nodes] = CLOSED_BOUNDARY
    status_at_node[0] = FIXED_VALUE_BOUNDARY
    assert_array_equal(model.grid.status_at_node, status_at_node)


def test_generic_bch(clock_simple, simple_square_grid):
    gfblh = GenericFuncBaselevelHandler(
        simple_square_grid,
        modify_core_nodes=True,
        function=lambda grid, t: -(grid.x_of_node + grid.y_of_node + (0 * t)),
    )
    params = {
        "grid": simple_square_grid,
        "clock": clock_simple,
        "boundary_handlers": {"mynew_bh": gfblh},
    }

    model = Basic(**params)
    bh = model.boundary_handlers["mynew_bh"]

    # assertion tests
    assert "mynew_bh" in model.boundary_handlers
    assert_array_equal(np.where(bh.nodes_to_lower)[0], model.grid.core_nodes)

    step = 10.0
    model.run_one_step(step)

    dzdt = -(model.grid.x_of_node + model.grid.y_of_node)
    truth_z = -1.0 * dzdt * step
    assert_array_equal(
        model.z[model.grid.core_nodes], truth_z[model.grid.core_nodes]
    )


def test_capture_node(clock_simple, simple_square_grid):
    cnblh = CaptureNodeBaselevelHandler(
        simple_square_grid,
        capture_node=1,
        capture_incision_rate=-3.0,
        capture_start_time=10,
        capture_stop_time=20,
        post_capture_incision_rate=-0.1,
    )

    params = {
        "grid": simple_square_grid,
        "clock": clock_simple,
        "boundary_handlers": {"CaptureNodeBaselevelHandler": cnblh},
    }

    model = Basic(**params)
    # assertion tests
    assert "CaptureNodeBaselevelHandler" in model.boundary_handlers
    assert model.z[1] == 0
    model.run_one_step(10.0)
    assert model.z[1] == 0
    model.run_one_step(10)
    assert model.z[1] == -30.0
    model.run_one_step(10)
    assert model.z[1] == -31.0
