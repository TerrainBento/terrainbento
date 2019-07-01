# coding: utf8
# !/usr/env/python

import os

import numpy as np
import pytest

from landlab import HexModelGrid, RasterModelGrid
from terrainbento.boundary_handlers import SingleNodeBaselevelHandler

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_hex():
    """Test using a hex grid."""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")

    bh = SingleNodeBaselevelHandler(mg, outlet_id=0, lowering_rate=-0.1)
    bh.run_one_step(10.0)

    assert z[1] == 0.0
    assert z[0] == -1.0


def test_passing_neither_lowering_method():
    """Test passing no lowering information."""
    mg = RasterModelGrid((5, 5))
    mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        SingleNodeBaselevelHandler(mg, outlet_id=0)


def test_passing_both_lowering_methods():
    """Test passing both lowering methods."""
    mg = RasterModelGrid((5, 5))
    mg.add_zeros("node", "topographic__elevation")
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")

    with pytest.raises(ValueError):
        SingleNodeBaselevelHandler(
            mg, outlet_id=0, lowering_rate=-0.1, lowering_file_path=file
        )


def test_outlet_lowering_object_bad_file():
    """Test using an outlet lowering object with a bad file."""

    mg = HexModelGrid(5, 5)
    mg.add_zeros("node", "topographic__elevation")

    with pytest.raises(ValueError):
        SingleNodeBaselevelHandler(
            mg, outlet_id=0, lowering_file_path="foo.txt"
        )


def test_outlet_lowering_rate_no_scaling_bedrock():
    """Test using an rate lowering object with no scaling and bedrock."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    node_id = 27
    bh = SingleNodeBaselevelHandler(mg, outlet_id=node_id, lowering_rate=-0.1)
    for _ in range(240):
        bh.run_one_step(10)

    assert z[1] == 1.0
    assert b[1] == 0.0

    assert z[node_id] == -239.0
    assert b[node_id] == -240.0


def test_outlet_lowering_rate_on_not_outlet():
    """Test using an rate lowering object with no scaling and bedrock."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    node_id = 27
    bh = SingleNodeBaselevelHandler(
        mg, outlet_id=node_id, lowering_rate=-0.1, modify_outlet_id=False
    )
    for _ in range(240):
        bh.run_one_step(10)

    assert z[node_id] == 1.0
    assert b[node_id] == 0.0

    not_outlet = mg.nodes != node_id
    assert np.all(z[not_outlet] == 241.0)
    assert np.all(b[not_outlet] == 240.0)


def test_outlet_lowering_object_no_scaling_bedrock():
    """Test using an outlet lowering object with no scaling and bedrock."""

    mg = HexModelGrid(5, 5)
    z = mg.add_ones("node", "topographic__elevation")
    b = mg.add_zeros("node", "bedrock__elevation")

    node_id = 27
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = SingleNodeBaselevelHandler(
        mg, outlet_id=node_id, lowering_file_path=file
    )
    for _ in range(241):
        bh.run_one_step(10)

    assert z[1] == 1.0
    assert b[1] == 0.0

    assert z[node_id] == -46.5
    assert b[node_id] == -47.5


def test_outlet_lowering_object_no_scaling():
    """Test using an outlet lowering object with no scaling."""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")
    node_id = 27
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = SingleNodeBaselevelHandler(
        mg, outlet_id=node_id, lowering_file_path=file
    )
    for _ in range(241):
        bh.run_one_step(10)

    assert z[1] == 0.0
    assert bh.z[node_id] == -47.5


def test_outlet_lowering_object_with_scaling():
    """Test using an outlet lowering object with scaling."""

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros("node", "topographic__elevation")
    node_id = 27
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    bh = SingleNodeBaselevelHandler(
        mg,
        outlet_id=node_id,
        lowering_file_path=file,
        model_end_elevation=-318.0,
    )

    for _ in range(241):
        bh.run_one_step(10)

    assert bh.z[node_id] == -95.0
    assert z[1] == 0.0


def test_outlet_lowering_modify_other_nodes():
    mg = HexModelGrid(5, 5)
    mg.add_zeros("node", "topographic__elevation")
    node_id = 27
    file = os.path.join(_TEST_DATA_DIR, "outlet_history.txt")
    with pytest.raises(ValueError):
        SingleNodeBaselevelHandler(
            mg,
            outlet_id=node_id,
            lowering_file_path=file,
            modify_outlet_id=False,
        )
