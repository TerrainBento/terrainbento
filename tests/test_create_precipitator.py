import os

import pytest

from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_bad_precipitator_params():
    filename = os.path.join(
        _TEST_DATA_DIR, "basic_inputs_bad_precipitator.yaml"
    )
    with pytest.raises(ValueError):
        ErosionModel.from_file(filename)


def test_bad_precipitator_instance(clock_simple, grid_1):
    not_a_precipitator = "I am not a precipitator"
    with pytest.raises(ValueError):
        ErosionModel(
            grid=grid_1, clock=clock_simple, precipitator=not_a_precipitator
        )
