import pytest

from terrainbento import (
    BasicSt,
    RandomPrecipitator,
    SimpleRunoff,
    UniformPrecipitator,
)


def test_not_UniformPrecipitator(grid_1, clock_simple):
    rp = RandomPrecipitator(grid_1)
    with pytest.raises(ValueError):
        BasicSt(clock_simple, grid_1, precipitator=rp)


def test_not_default_UniformPrecipitator(grid_1, clock_simple):
    rp = UniformPrecipitator(grid_1, rainfall_flux=2.0)
    with pytest.raises(ValueError):
        BasicSt(clock_simple, grid_1, precipitator=rp)


def test_not_SimpleRunoff(grid_1, clock_simple):
    rg = RandomPrecipitator(grid_1)
    with pytest.raises(ValueError):
        BasicSt(clock_simple, grid_1, runoff_generator=rg)


def test_not_default_default_SimpleRunoff(grid_1, clock_simple):
    grid_1.add_ones("node", "rainfall__flux")
    rg = SimpleRunoff(grid_1, runoff_proportion=0.7)
    with pytest.raises(ValueError):
        BasicSt(clock_simple, grid_1, runoff_generator=rg)
