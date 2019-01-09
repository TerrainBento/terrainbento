import pytest

from terrainbento import BasicStVs


def test_bad_transmiss(grid_2, clock_simple):
    params = {
        "grid": grid_2,
        "clock": clock_simple,
        "hydraulic_conductivity": 0.,
    }

    with pytest.raises(ValueError):
        BasicStVs(**params)
