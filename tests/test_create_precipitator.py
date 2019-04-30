import pytest

from terrainbento import ErosionModel


def test_bad_precipitator_params(tmpdir, basic_inputs_bad_precipitator_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_bad_precipitator_yaml)
        with pytest.raises(ValueError):
            ErosionModel.from_file("./params.yaml")


def test_bad_precipitator_instance(clock_simple, grid_1):
    not_a_precipitator = "I am not a precipitator"
    with pytest.raises(ValueError):
        ErosionModel(
            grid=grid_1, clock=clock_simple, precipitator=not_a_precipitator
        )
