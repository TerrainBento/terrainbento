# coding: utf8
# !/usr/env/python

import os

import pytest

from terrainbento import Clock

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_from_file():
    filename = os.path.join(_TEST_DATA_DIR, "clock.yaml")
    clock = Clock.from_file(filename)
    assert clock.start == 1.
    assert clock.stop == 11.
    assert clock.step == 2.


@pytest.mark.parametrize("bad", ["start", "stop", "step"])
def test_bad_values(bad):
    params = {"start": 0, "stop": 10, "step": 2}
    params[bad] = "spam"
    with pytest.raises(ValueError):
        Clock.from_dict(params)


def test_start_larger():
    with pytest.raises(ValueError):
        Clock(stop=-1, step=1)
