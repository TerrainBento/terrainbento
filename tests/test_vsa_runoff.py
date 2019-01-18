import pytest

from landlab import RasterModelGrid
from terrainbento import VariableSourceAreaRunoff


def test_add_water_unit_flux_in():
    grid = RasterModelGrid((5, 5))
    VariableSourceAreaRunoff(grid)
    assert "water__unit_flux_in" in grid.at_node


def test_negative_hydraulic_conductivity(grid_2, clock_simple):
    with pytest.raises(ValueError):
        VariableSourceAreaRunoff(grid_2, hydraulic_conductivity=-1.0)


def test_negative_trasmissivity(grid_2, clock_simple):
    grid_2.at_node["soil__depth"][:] = -10.0
    grid_2.add_ones("node", "rainfall__flux")
    rg = VariableSourceAreaRunoff(grid_2, hydraulic_conductivity=1.0)

    with pytest.raises(ValueError):
        rg.run_one_step(0)
