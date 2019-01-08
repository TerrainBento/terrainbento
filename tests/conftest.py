import pytest

from landlab import RasterModelGrid
from terrainbento import Clock


@pytest.fixture()
def U():
    U = 0.0001
    return U

@pytest.fixture()
def K():
    K = 0.01
    return K

@pytest.fixture()
def Kr():
    Kr = 0.001
    return Kr

@pytest.fixture()
def Kt():
    Kt = 0.005
    return Kt

@pytest.fixture()
def grid_1():
    grid = RasterModelGrid((3, 21), xy_spacing=100.)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_zeros("node", "soil__depth")
    return grid


@pytest.fixture()
def grid_2():
    grid = RasterModelGrid((8, 20), xy_spacing=100.)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    grid.add_zeros("node", "soil__depth")
    lith = grid.add_zeros("node", "lithology_contact__elevation")
    lith[:59] = -10000.
    lith[60:] = 10
    return grid

@pytest.fixture()
def clock_simple():
    clock_simple = Clock(step=1000., stop=5.1e6)
    return clock_simple


@pytest.fixture()
def clock_01():
    clock_01 = {"step": 1., "stop": 10.}
    return clock_01


@pytest.fixture()
def clock_02():
    clock_02 = {"step": 10., "stop": 1000.}
    return clock_02


@pytest.fixture()
def clock_03():
    clock_03 = {"step": 10., "stop": 1e6}
    return clock_03


@pytest.fixture()
def clock_04():
    clock_04 = {"step": 10., "stop": 100000.}
    return clock_04


@pytest.fixture()
def clock_05():
    clock_05 = {"step": 10., "stop": 200.}
    return clock_05


@pytest.fixture()
def clock_06():
    clock_06 = {"step": 1., "stop": 3.}
    return clock_06


@pytest.fixture()
def clock_07():
    clock_07 = {"step": 10., "stop": 10000.}
    return clock_07


@pytest.fixture()
def clock_08():
    clock_08 = {"step": 1., "stop": 20.}
    return clock_08


@pytest.fixture()
def clock_09():
    clock_09 = {"step": 2, "stop": 200.}
    return clock_09


@pytest.fixture()
def precip_defaults():
    precip_defaults = {
        "daily_rainfall__intermittency_factor": 0.5,
        "daily_rainfall__intermittency_factor_time_rate_of_change": 0.1,
        "rainfall__mean_rate": 1.0,
        "rainfall__mean_rate_time_rate_of_change": 0.2,
        "infiltration_capacity": 0,
        "rainfall__shape_factor": 0.65,
    }
    return precip_defaults


@pytest.fixture()
def precip_testing_factor():
    precip_testing_factor = 1.3145341380253433
    return precip_testing_factor
