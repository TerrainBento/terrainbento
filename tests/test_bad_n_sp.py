import pytest

from terrainbento import (
    BasicChRtTh,
    BasicDd,
    BasicDdRt,
    BasicDdSt,
    BasicDdVs,
    BasicRtTh,
    BasicStTh,
    BasicTh,
    BasicThVs,
)


@pytest.mark.parametrize(
    "Model",
    [
        BasicDd,
        BasicDdRt,
        BasicDdRt,
        BasicDdVs,
        BasicRtTh,
        BasicStTh,
        BasicTh,
        BasicThVs,
        BasicDdSt,
        BasicChRtTh,
    ],
)
def test_bad_n_sp(clock_simple, grid_1, Model):
    params = {"grid": grid_1, "clock": clock_simple, "n_sp": 1.01}

    with pytest.raises(ValueError):
        Model(**params)
