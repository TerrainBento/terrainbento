# coding: utf8
# !/usr/env/python

import pytest

from terrainbento import Clock


def test_from_file(tmpdir, clock_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(clock_yaml)
        clock = Clock.from_file("./params.yaml")
    assert clock.start == 1.0
    assert clock.stop == 11.0
    assert clock.step == 2.0


@pytest.mark.parametrize("bad", ["start", "stop", "step"])
def test_bad_values(bad):
    params = {"start": 0, "stop": 10, "step": 2}
    params[bad] = "spam"
    with pytest.raises(ValueError):
        Clock.from_dict(params)


def test_start_larger():
    with pytest.raises(ValueError):
        Clock(stop=-1, step=1)
