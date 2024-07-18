# !/usr/env/python

# This file has tests for the new style output writers.

import glob
import itertools
import os

import numpy as np
import pytest

from terrainbento import Basic, NotCoreNodeBaselevelHandler
from terrainbento.output_writers import (
    GenericOutputWriter,
    OutputIteratorSkipWarning,
    OWSimpleNetCDF,
    StaticIntervalOutputWriter,
)
from terrainbento.utilities import filecmp

_TEST_OUTPUT_DIR = os.path.join(os.curdir, "output")
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_output_filepath(filename):
    return os.path.join(_TEST_OUTPUT_DIR, filename)


def cleanup_files(searchpath):
    files = glob.glob(searchpath)
    for f in files:
        os.remove(f)


def fibonnaci():
    """Yields a fibonacci sequence."""
    yield 0.0
    yield 1.0
    a, b = 0.0, 1.0
    while True:
        a, b = b, a + b
        yield b


class OWStaticWrapper(StaticIntervalOutputWriter):
    """Wraps the StaticIntervalOutputWriter for testings."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        average_change = np.mean(self.change[self.model.grid.core_nodes])
        filepath = get_output_filepath(f"{self.filename_prefix}.txt")
        with open(filepath, "w") as f:
            f.write(str(average_change))
        self.register_output_filepath(filepath)


class OWGenericWrapper(GenericOutputWriter):
    """Wraps the GenericIntervalOutputWriter for testings."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        average_change = np.mean(self.change[self.model.grid.core_nodes])
        filepath = get_output_filepath(f"{self.filename_prefix}.txt")
        with open(filepath, "w") as f:
            f.write(str(average_change))
        self.register_output_filepath(filepath)


#  To test:
# - Try to break model time passing next output time? (assertion in
#   ErosionModel.write_output() I don't know how to get it to fail though.


def test_out_of_phase_interval(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    interval = 5.5
    warning_msg_sample = "time that is not divisible by the model step"
    with pytest.warns(UserWarning, match=warning_msg_sample):
        model = Basic(
            clock_08,
            almost_default_grid,
            water_erodibility=0.0,
            regolith_transport_parameter=0.0,
            boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
            output_writers={
                "out-of-phase-ow": {
                    "class": OWStaticWrapper,
                    "kwargs": {
                        "add_id": False,
                        "intervals": interval,
                    },
                },
            },
            # output_interval=6.0,
            output_dir=_TEST_OUTPUT_DIR,
            output_prefix="",
            save_first_timestep=True,
            save_last_timestep=True,
        )
        model.run()

    # The model should still run fine, but any out of phase output times are
    # delayed to the next step.
    for time_int in [0, 6, 11, 17, 20]:  # instead of [0, 5.5, 11, 16.5, 20]
        # assert things were done correctly
        filename = f"out-of-phase-ow_time-{float(time_int):012.1f}.txt"
        truth_file = os.path.join(_TEST_DATA_DIR, f"truth_{filename}")
        test_file = get_output_filepath(filename)
        assert filecmp(test_file, truth_file) is True

    model.remove_output()


def test_out_of_phase_interval_last(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    interval = 19.5
    warning_msg_sample = "time that is not divisible by the model step"
    with pytest.warns(UserWarning, match=warning_msg_sample):
        model = Basic(
            clock_08,
            almost_default_grid,
            water_erodibility=0.0,
            regolith_transport_parameter=0.0,
            boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
            output_writers={
                "out-of-phase-ow": {
                    "class": OWStaticWrapper,
                    "kwargs": {
                        "add_id": False,
                        "intervals": interval,
                    },
                },
            },
            # output_interval=6.0,
            output_dir=_TEST_OUTPUT_DIR,
            output_prefix="",
            save_first_timestep=True,
            save_last_timestep=True,
        )
        model.run()

    # The model should still run fine, but any out of phase output times are
    # delayed to the next step.

    # Check that there is no file for t=19.5
    bad_filename = f"out-of-phase-ow_time-{19.5:012.1f}.txt"
    bad_filepath = get_output_filepath(bad_filename)
    assert not os.path.isfile(bad_filepath), f"{bad_filepath} should not exist"

    # Check that the output that should exist
    for time_int in [0, 20]:  # instead of [0, 19.5, 20]
        filename = f"out-of-phase-ow_time-{float(time_int):012.1f}.txt"
        truth_file = os.path.join(_TEST_DATA_DIR, f"truth_{filename}")
        test_file = get_output_filepath(filename)
        assert filecmp(test_file, truth_file) is True

    model.remove_output()


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
def test_out_of_phase_interval_warns(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    times = [
        0.0,
        1.0,
        1.01,  # next time is set to 2. Divisible warning
        1.02,  # new time < 2. Starting skips. Skipping warning
        1.10,
        1.11,
        1.12,
        1.13,
        1.14,
        1.15,
        1.16,
        1.17,
        1.18,
        3.00,  # Finds suitable time on 10th skip.
    ]
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "out-of-phase-skipping-ow": {
                "class": OWStaticWrapper,
                "kwargs": {
                    "add_id": False,
                    "times": times,
                },
            },
        },
        # output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        save_first_timestep=True,
        save_last_timestep=True,
    )
    skip_warning_msg = "next time that is less than or equal to the current"
    divisible_warning_msg = "time that is not divisible by the model step"

    # Copied from ErosionModel.run() so I can control the steps
    model._itters = []
    if model.save_first_timestep:
        model.iteration = 0
        model._itters.append(0)
        model.write_output()
    model.iteration = 1

    def run_one_iteration():
        time_now = model._model_time
        next_run_pause = min(
            model.next_output_time,
            model.clock.stop,
        )
        assert next_run_pause > time_now
        print(f"Iteration {model.iteration:05d} Model time {model.model_time}")
        print(f"  Run for {next_run_pause - time_now}, (step = {model.clock.step})")
        model.run_for(model.clock.step, next_run_pause - time_now)
        time_now = model._model_time
        model._itters.append(model.iteration)
        model.write_output()
        model.iteration += 1

    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 0.0 and model.next_output_time == 1.0

    warning_msg_sample = "time that is not divisible by the model step"
    with pytest.warns(UserWarning, match=warning_msg_sample):
        run_one_iteration()
    # times_iter returns 1.01, which is then delayed to 2.0
    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 1.00 and model.next_output_time == 2.0

    # time_iter returns 1.02, which triggers skips but eventually finds 3.0
    with pytest.warns(UserWarning) as all_warnings:
        run_one_iteration()
    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 2.0 and model.next_output_time == 3.0
    ignored_warnings = [RuntimeWarning]
    for warn_info in all_warnings:
        warn_type = warn_info.category
        if warn_type in ignored_warnings:
            continue
        try:
            assert warn_type == UserWarning
            message = warn_info.message.args[0]
            is_skip_warning = skip_warning_msg in message
            is_divisible_warning = divisible_warning_msg in message
            assert is_skip_warning or is_divisible_warning
        except AssertionError:
            print(warn_info)  # So I can see other warnings
            raise

    run_one_iteration()
    # times_iter returns 20.0
    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 3.0 and model.next_output_time == 20.0

    run_one_iteration()
    # Model finishes
    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 20.0 and model.next_output_time == np.inf

    # now that the model is finished running, execute finalize.
    model.finalize()

    # Check that there is a file for t=3.0
    good_filename = f"out-of-phase-skipping-ow_time-{3.0:012.1f}.txt"
    good_filepath = get_output_filepath(good_filename)
    assert os.path.isfile(good_filepath), f"{good_filepath} should exist"

    model.remove_output()


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
def test_out_of_phase_interval_fails(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    times = [
        0.0,
        1.0,
        1.01,  # next time is set to 2. Divisible warning
        1.02,  # new time < 2. Starting skips. Skipping warning
        1.10,
        1.11,
        1.12,
        1.13,
        1.14,
        1.15,
        1.16,
        1.17,
        1.18,
        2.00,  # 10 skips. Loop exits. Raises Assertion Error
        3.0,  # Does not get this far
    ]
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "out-of-phase-ow-fails": {
                "class": OWStaticWrapper,
                "kwargs": {
                    "add_id": False,
                    "times": times,
                },
            },
        },
        # output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        save_first_timestep=True,
        save_last_timestep=True,
    )
    skip_warning_msg = "next time that is less than or equal to the current"
    divisible_warning_msg = "time that is not divisible by the model step"

    # Copied from ErosionModel.run() so I can control the steps
    model._itters = []

    if model.save_first_timestep:
        model.iteration = 0
        model._itters.append(0)
        model.write_output()
    model.iteration = 1

    def run_one_iteration():
        time_now = model._model_time
        next_run_pause = min(
            # time_now + model.output_interval, model.clock.stop,
            model.next_output_time,
            model.clock.stop,
        )
        assert next_run_pause > time_now
        print(f"Iteration {model.iteration:05d} Model time {model.model_time}")
        print(f"  Run for {next_run_pause - time_now}, (step = {model.clock.step})")
        model.run_for(model.clock.step, next_run_pause - time_now)
        time_now = model._model_time
        model._itters.append(model.iteration)
        model.write_output()
        model.iteration += 1

    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 0.0 and model.next_output_time == 1.0

    warning_msg_sample = "time that is not divisible by the model step"
    with pytest.warns(UserWarning, match=warning_msg_sample):
        run_one_iteration()
    # times_iter returns 1.01, which is then delayed to 2.0
    print(f"t={model.model_time}, nt={model.next_output_time}")
    assert model.model_time == 1.00 and model.next_output_time == 2.0

    # time_iter returns 1.02, which triggers skips and eventually fails
    with pytest.warns(UserWarning) as all_warnings, pytest.raises(AssertionError):
        run_one_iteration()
    ignored_warnings = [RuntimeWarning]
    for warn_info in all_warnings:
        warn_type = warn_info.category
        if warn_type in ignored_warnings:
            continue
        try:
            assert warn_type == UserWarning
            message = warn_info.message.args[0]
            is_skip_warning = skip_warning_msg in message
            is_divisible_warning = divisible_warning_msg in message
            assert is_skip_warning or is_divisible_warning
        except AssertionError:
            print(warn_info)  # So I can see other warnings
            raise

    model.remove_output()


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
def test_multiple_frequencies(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    common_interval = 2.0
    uncommon_interval = 5.0
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "common-ow": {
                "class": OWStaticWrapper,
                "kwargs": {
                    "add_id": False,
                    "intervals": common_interval,
                },
            },
            "uncommon-ow": {
                "class": OWStaticWrapper,
                "kwargs": {
                    "add_id": False,
                    "intervals": uncommon_interval,
                },
            },
        },
        # output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        save_first_timestep=True,
        save_last_timestep=True,
    )
    model.run()

    for name, interval in [
        ("common-ow", common_interval),
        ("uncommon-ow", uncommon_interval),
    ]:
        for output_time in itertools.count(0.0, interval):
            if output_time > clock_08.stop:
                # Break the infinite iterator at the clock stop time
                break
            # assert things were done correctly
            filename = f"{name}_time-{output_time:012.1f}.txt"
            truth_file = os.path.join(_TEST_DATA_DIR, f"truth_{filename}")
            test_file = get_output_filepath(filename)
            assert filecmp(test_file, truth_file) is True

    model.remove_output()


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
def test_custom_iter(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "fibonnaci": {
                "class": OWGenericWrapper,
                "kwargs": {
                    "add_id": False,
                    "times_iter": fibonnaci(),
                },
            },
        },
        # output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        save_first_timestep=True,
        save_last_timestep=True,
    )
    with pytest.warns(OutputIteratorSkipWarning):
        model.run()

    for time_int in [0, 1, 2, 3, 5, 8, 13, 20]:
        # Note: the second 1 in the fib sequence will be skipped

        # assert things were done correctly
        filename = f"fibonnaci_time-{float(time_int):012.1f}.txt"
        truth_file = os.path.join(_TEST_DATA_DIR, f"truth_{filename}")
        test_file = get_output_filepath(filename)
        assert filecmp(test_file, truth_file) is True

    model.remove_output()


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
def test_deleting_output(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "fibonnaci-1": {
                "class": OWGenericWrapper,
                "kwargs": {
                    "add_id": False,
                    "times_iter": fibonnaci(),
                    "verbose": True,
                },
            },
            "fibonnaci-2": {
                "class": OWGenericWrapper,
                "kwargs": {
                    "add_id": False,
                    "times_iter": fibonnaci(),
                    "verbose": True,
                },
            },
            "fibonnaci-3": {
                "class": OWGenericWrapper,
                "kwargs": {
                    "add_id": False,
                    "times_iter": fibonnaci(),
                    "verbose": True,
                },
            },
            "netcdf-1": {
                "class": OWSimpleNetCDF,
                "args": ["topographic__elevation"],
                "kwargs": {"intervals": 5.0, "add_id": False, "verbose": True},
            },
            "netcdf-2": {
                "class": OWSimpleNetCDF,
                "args": ["topographic__elevation"],
                "kwargs": {"intervals": 5.0, "add_id": False, "verbose": True},
            },
            "netcdf-3": {
                "class": OWSimpleNetCDF,
                "args": ["topographic__elevation"],
                "kwargs": {"intervals": 5.0, "add_id": False, "verbose": True},
            },
        },
        # output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        output_default_netcdf=False,
        save_first_timestep=True,
        save_last_timestep=True,
    )
    with pytest.warns(OutputIteratorSkipWarning):
        model.run()

    class NotAWriter:
        pass

    not_a_writer = NotAWriter()
    with pytest.raises(TypeError):
        model.remove_output(writer=not_a_writer)
    with pytest.raises(TypeError):
        model.remove_output(extension=not_a_writer)

    def all_exist(filepaths):
        assert all([os.path.isfile(fp) for fp in filepaths])

    def all_deleted(filepaths):
        assert all([not os.path.isfile(fp) for fp in filepaths])

    ow_fib_1 = model.get_output_writer("fibonnaci-1")[0]
    ow_fib_2 = model.get_output_writer("fibonnaci-2")[0]
    ow_fib_3 = model.get_output_writer("fibonnaci-3")[0]
    ow_nc_all = model.get_output_writer("netcdf-")
    ow_nc_1, ow_nc_2, ow_nc_3 = ow_nc_all
    assert ow_fib_1 and ow_fib_1.name == "fibonnaci-1"
    assert ow_fib_2 and ow_fib_2.name == "fibonnaci-2"
    assert ow_fib_3 and ow_fib_3.name == "fibonnaci-3"
    assert ow_nc_1 and ow_nc_1.name == "netcdf-1"
    assert ow_nc_2 and ow_nc_2.name == "netcdf-2"
    assert ow_nc_3 and ow_nc_3.name == "netcdf-3"

    out_fib_1 = model.get_output(writer=ow_fib_1)
    out_fib_2 = model.get_output(writer="fibonnaci-2")
    out_fib_3 = model.get_output(writer=["fibonnaci-3"])
    out_nc_1 = model.get_output(writer=ow_nc_1)
    out_nc_2 = model.get_output(writer=ow_nc_2)
    out_nc_3 = model.get_output(writer=ow_nc_3)
    out_nc_2_and_3 = model.get_output(writer=["netcdf-2", "netcdf-3"])
    out_nc_all = model.get_output(writer=ow_nc_all)
    assert (out_nc_1 + out_nc_2 + out_nc_3) == out_nc_all
    assert (out_nc_1 + out_nc_2_and_3) == out_nc_all
    assert model.get_output(extension="txt", writer=ow_nc_all) == []
    all_exist(out_fib_1 + out_fib_2 + out_fib_3)
    all_exist(out_nc_1 + out_nc_2 + out_nc_3)

    # Attempt to delete a file type that the writer never made
    model.remove_output(extension=".pdf", writer=None)
    model.remove_output(extension="pdf", writer=None)
    model.remove_output(extension="pdf", writer=ow_fib_1)
    model.remove_output(extension="pdf", writer=[ow_fib_1])
    all_exist(out_fib_1 + out_fib_2 + out_fib_3)
    all_exist(out_nc_1 + out_nc_2 + out_nc_3)

    # Delete with variations of the writer arg
    model.remove_output(extension=".txt", writer=ow_fib_1)  # test the period
    all_exist(out_fib_2 + out_fib_3)
    all_deleted(out_fib_1)
    model.remove_output(extension="txt", writer=[ow_fib_2])
    all_exist(out_fib_3)
    all_deleted(out_fib_1 + out_fib_2)
    model.remove_output(extension="txt", writer=None)
    all_deleted(out_fib_1 + out_fib_2 + out_fib_3)
    all_exist(out_nc_1 + out_nc_2 + out_nc_3)

    # Delete with variations of the extension arg
    model.remove_output(extension="nc", writer=ow_nc_1)
    all_exist(out_nc_2 + out_nc_3)
    all_deleted(out_nc_1)
    model.remove_output(extension=["nc"], writer=ow_nc_2)
    all_exist(out_nc_3)
    all_deleted(out_nc_1 + out_nc_2)
    model.remove_output(extension=None, writer=ow_nc_3)
    all_deleted(out_nc_1 + out_nc_2 + out_nc_3)


@pytest.mark.parametrize(
    "times, intervals, correct_times",
    [
        (None, None, [0, 6, 12, 18, 20, None]),
        ([0, 1, 5], None, [0, 1, 5, 20, None]),
        (None, 7.0, [0, 7, 14, 20, None]),
    ],
)
def test_static_default(
    clock_08,
    almost_default_grid,
    times,
    intervals,
    correct_times,
):
    static_kwargs = {
        "add_id": False,
        "save_first_timestep": True,
        "save_last_timestep": True,
    }
    if times is not None:
        static_kwargs["times"] = times

    if intervals is not None:
        static_kwargs["intervals"] = intervals

    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )
    model = Basic(
        clock_08,
        almost_default_grid,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "out-of-phase-ow": {
                "class": OWStaticWrapper,
                "kwargs": static_kwargs,
            },
        },
        output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        output_prefix="",
        save_first_timestep=True,
        save_last_timestep=True,
    )

    static_ow = model.get_output_writer("out-of-phase-ow")[0]
    for t_int in correct_times:
        if t_int is None:
            print(f"checking: {t_int} is {static_ow.next_output_time}")
            assert static_ow.next_output_time is None
        else:
            print(f"checking: {float(t_int)} == {static_ow.next_output_time}")
            assert static_ow.next_output_time == float(t_int)
        static_ow.advance_iter()

    model.remove_output()
