import pytest

from terrainbento import BasicDdHy, BasicHy, BasicHyRt, BasicHySt, BasicHyVs


@pytest.mark.parametrize(
    "Model", [BasicHy, BasicHySt, BasicDdHy, BasicHyVs, BasicHyRt]
)
def test_porosity_error(
    clock_simple, grid_1, Model,
):
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "sediment_porosity": 0.3,
    }

    with pytest.raises(ValueError):
        Model(**params)
