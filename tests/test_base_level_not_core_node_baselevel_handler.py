# coding: utf8
# !/usr/env/python

import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from landlab import HexModelGrid, RasterModelGrid
from terrainbento.boundary_handlers import NotCoreNodeBaselevelHandler

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_hex():
    """Test using a hex grid."""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")

    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=False, lowering_rate=-0.1
    )
    bh.run_one_step(10.0)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # closed should have been downdropped 10*0.1
    assert_array_equal(z[closed], -1.0 * np.ones(np.sum(closed)))

    # not closed should have stayed the same
    assert_array_equal(z[not_closed], np.zeros(np.sum(not_closed)))


def test_passing_neither_lowering_method():
    """Test passing no lowering information."""
    mg = RasterModelGrid((5, 5))
    mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        NotCoreNodeBaselevelHandler(mg)


def test_passing_both_lowering_methods():
    """Test passing both lowering methods."""
    mg = RasterModelGrid((5, 5))
    mg.add_zeros("node", "topographic__elevation")
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")

    with pytest.raises(ValueError):
        NotCoreNodeBaselevelHandler(
            mg, lowering_rate=-0.1, lowering_file_path=file
        )


def test_outlet_lowering_object_bad_file():
    """Test using an outlet lowering object with a bad file."""

    mg = HexModelGrid(5, 5)
    mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        NotCoreNodeBaselevelHandler(mg, lowering_file_path="foo.txt")


def test_outlet_lowering_rate_no_scaling_bedrock():
    """Test using an outlet lowering rate with no scaling and bedrock."""

    mg = RasterModelGrid((5, 5))
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=True, lowering_rate=-0.1
    )
    for _ in range(240):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # closed should have stayed the same
    assert_array_equal(z[closed], np.ones(np.sum(closed)))
    assert_array_equal(b[closed], np.zeros(np.sum(closed)))

    # not closed should have been uplifted 2410*0.1
    assert_array_equal(b[not_closed], 240.0 * np.ones(np.sum(not_closed)))
    assert_array_equal(z[not_closed], 241.0 * np.ones(np.sum(not_closed)))

    # % doing the oposite should also work
    mg = RasterModelGrid((5, 5))
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=False, lowering_rate=-0.1
    )
    for _ in range(240):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # not closed should have staued the same
    assert_array_equal(z[not_closed], np.ones(np.sum(not_closed)))
    assert_array_equal(b[not_closed], np.zeros(np.sum(not_closed)))

    # closed should have lowered by 240
    assert_array_equal(b[closed], -240.0 * np.ones(np.sum(closed)))
    assert_array_equal(z[closed], -239.0 * np.ones(np.sum(closed)))


def test_outlet_lowering_object_no_scaling_bedrock():
    """Test using an outlet lowering object with no scaling and bedrock."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=False, lowering_file_path=file
    )
    for _ in range(241):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # closed should have lowered -46.5
    assert_array_equal(z[closed], -46.5 * np.ones(np.sum(closed)))
    assert_array_equal(b[closed], -47.5 * np.ones(np.sum(closed)))

    # not closed should stayed the same
    assert_array_equal(z[not_closed], np.ones(np.sum(not_closed)))
    assert_array_equal(b[not_closed], np.zeros(np.sum(not_closed)))


def test_outlet_lowering_object_no_scaling():
    """Test using an outlet lowering object with no scaling."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=False, lowering_file_path=file
    )
    for _ in range(241):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # not closed should have stayed the same
    assert_array_equal(z[not_closed], np.ones(np.sum(not_closed)))

    # closed should lowered by 47.5  to -46.5
    assert_array_equal(z[closed], -46.5 * np.ones(np.sum(closed)))


def test_outlet_lowering_object_no_scaling_core_nodes():
    """Test using an outlet lowering object with no scaling on core nodes."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = NotCoreNodeBaselevelHandler(
        mg, modify_core_nodes=True, lowering_file_path=file
    )
    for _ in range(241):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # closed should have stayed the same
    assert_array_equal(z[closed], np.ones(np.sum(closed)))

    # not closed should raise by 47.5  to 48.5
    assert_array_almost_equal(
        z[not_closed], 48.5 * np.ones(np.sum(not_closed))
    )


def test_outlet_lowering_object_with_scaling():
    """Test using an outlet lowering object with scaling."""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = NotCoreNodeBaselevelHandler(
        mg,
        modify_core_nodes=False,
        lowering_file_path=file,
        model_end_elevation=-318.0,
    )
    for _ in range(241):
        bh.run_one_step(10)

    closed = mg.status_at_node != 0
    not_closed = mg.status_at_node == 0

    # closed should have lowered -46.5
    assert_array_equal(z[closed], -95.0 * np.ones(np.sum(closed)))

    # not closed should stayed the same
    assert_array_equal(z[not_closed], np.zeros(np.sum(not_closed)))
