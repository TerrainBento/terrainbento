# coding: utf8
# !/usr/env/python

from landlab import HexModelGrid, RasterModelGrid
from terrainbento.boundary_handlers import CaptureNodeBaselevelHandler


def test_hex():
    """Test using a hex grid."""

    mg = HexModelGrid(5, 5)
    mg.add_zeros("node", "topographic__elevation")

    bh = CaptureNodeBaselevelHandler(
        mg,
        capture_node=3,
        capture_incision_rate=-3.0,
        capture_start_time=10,
        capture_stop_time=20,
        post_capture_incision_rate=-0.1,
    )
    bh.run_one_step(10)


def test_no_stop_time():
    """Test with no stop time."""

    mg = RasterModelGrid((5, 5))
    z = mg.add_zeros("node", "topographic__elevation")

    bh = CaptureNodeBaselevelHandler(
        mg, capture_node=3, capture_incision_rate=-3.0, capture_start_time=0
    )

    for _ in range(10):
        bh.run_one_step(10)

    assert z[3] == -3.0 * 10 * 10
