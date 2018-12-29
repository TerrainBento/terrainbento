import pytest


@pytest.fixture()
def clock_simple():
    clock_simple = {"dt": 1., "output_interval": 2., "run_duration": 200.}
    return clock_simple


@pytest.fixture()
def clock_01():
    clock_01 = {"dt": 1., "output_interval": 2., "run_duration": 10.}
    return clock_01


@pytest.fixture()
def clock_02():
    clock_02 = {"dt": 10., "output_interval": 2., "run_duration": 1000.}
    return clock_02


@pytest.fixture()
def clock_03():
    clock_03 = {"dt": 10., "output_interval": 1e3, "run_duration": 1e6}
    return clock_03


@pytest.fixture()
def clock_04():
    clock_04 = {"dt": 10., "output_interval": 2., "run_duration": 100000.}
    return clock_04


@pytest.fixture()
def clock_05():
    clock_05 = {"dt": 10., "output_interval": 2., "run_duration": 200.}
    return clock_05


@pytest.fixture()
def clock_06():
    clock_06 = {"dt": 1., "output_interval": 2., "run_duration": 3.}
    return clock_06


@pytest.fixture()
def clock_07():
    clock_07 = {"dt": 10., "output_interval": 2., "run_duration": 10000.}
    return clock_07


@pytest.fixture()
def clock_08():
    clock_08 = {"dt": 1., "output_interval": 20., "run_duration": 20.}
    return clock_08


@pytest.fixture()
def clock_09():
    clock_09 = {"dt": 2, "output_interval": 2., "run_duration": 200.}
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
