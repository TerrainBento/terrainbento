import os

import pytest

from terrainbento import ErosionModel

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_bad_runoff_instance(clock_simple, grid_1):
    not_a_runoff_generator = "I am not a runoff_generator"
    with pytest.raises(ValueError):
        ErosionModel(
            grid=grid_1,
            clock=clock_simple,
            runoff_generator=not_a_runoff_generator,
        )
