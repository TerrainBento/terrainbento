# coding: utf8
# !/usr/env/python

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from terrainbento.base_class import TwoLithologyErosionModel


def test_no_contact_zone_width(clock_simple, grid_5):

    params = {"grid": grid_5, "clock": clock_simple, "contact_zone__width": 0}

    model = TwoLithologyErosionModel(**params)
    model._setup_rock_and_till()

    truth = np.ones(model.grid.size("node"))
    truth[model.grid.core_nodes[14:]] = 0.0

    assert_array_equal(model.erody_wt, truth)


def test_contact_zone_width(clock_simple, grid_5):
    params = {
        "grid": grid_5,
        "clock": clock_simple,
        "contact_zone__width": 10.0,
    }

    model = TwoLithologyErosionModel(**params)
    model._setup_rock_and_till()

    truth = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.0,
            0.0,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.95257413,
            0.0,
            0.0,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.0,
            0.0,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.26894142,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert_array_almost_equal(model.erody_wt, truth)
