# !/usr/env/python

# This file has tests for the old style output writers to ensure backwards
# compatibility. All of the existing tests for output writers are kept as is.
# There are a few new ones too.

import glob
import os

import numpy as np

from terrainbento import Basic, NotCoreNodeBaselevelHandler
from terrainbento.utilities import filecmp

_TEST_OUTPUT_DIR = os.path.join(os.curdir, "output")
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_output_filepath(filename):
    return os.path.join(_TEST_OUTPUT_DIR, filename)


def cleanup_files(searchpath):
    files = glob.glob(searchpath)
    for f in files:
        os.remove(f)


# Some output writers
def output_writer_function_a(model):
    average_elevation = np.mean(model.z[model.grid.core_nodes])

    filepath = get_output_filepath(f"ow_func_a.{str(model.model_time)}.txt")
    with open(filepath, "w") as f:
        f.write(str(average_elevation))


def output_writer_function_b(model):
    minimum_elevation = np.min(model.z[model.grid.core_nodes])
    filepath = get_output_filepath(f"ow_func_b.{str(model.model_time)}.txt")
    with open(filepath, "w") as f:
        f.write(str(minimum_elevation))


class output_writer_class_a:
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        average_change = np.mean(self.change[self.model.grid.core_nodes])
        model_time_str = str(self.model.model_time)
        filepath = get_output_filepath(f"ow_class_a.{model_time_str}.txt")
        with open(filepath, "w") as f:
            f.write(str(average_change))


class output_writer_class_b:
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        min_change = np.min(self.change[self.model.grid.core_nodes])
        model_time_str = str(self.model.model_time)
        filepath = get_output_filepath(f"ow_class_b.{model_time_str}.txt")
        with open(filepath, "w") as f:
            f.write(str(min_change))


# Unchanged tests
# These tests should have minimal changes to ensure backwards compatibility
# I only changed where output files are saved (because failed tests don't clean
# up so they fill my test directory with junk files)


def test_one_function_writer(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )
    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        save_first_timestep=False,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={"function": [output_writer_function_a]},
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    test_file = get_output_filepath("ow_func_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_a.*.txt")


def test_one_class_writer(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )
    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        save_first_timestep=False,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={"class": [output_writer_class_a]},
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    test_file = get_output_filepath("ow_class_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_class_a.*.txt")


def test_two_function_writers(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )
    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        save_first_timestep=False,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "function": [output_writer_function_a, output_writer_function_b]
        },
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    test_file = get_output_filepath("ow_func_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    test_file = get_output_filepath("ow_func_b.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_*.txt")


def test_two_class_writers(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )
    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        save_first_timestep=False,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={"class": [output_writer_class_a, output_writer_class_b]},
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    test_file = get_output_filepath("ow_class_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    test_file = get_output_filepath("ow_class_b.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_class_*.txt")


def test_all_four_writers(clock_08, almost_default_grid):
    ncnblh = NotCoreNodeBaselevelHandler(
        almost_default_grid, modify_core_nodes=True, lowering_rate=-1
    )

    # construct and run model
    model = Basic(
        clock_08,
        almost_default_grid,
        save_first_timestep=False,
        water_erodibility=0.0,
        regolith_transport_parameter=0.0,
        boundary_handlers={"NotCoreNodeBaselevelHandler": ncnblh},
        output_writers={
            "function": [output_writer_function_a, output_writer_function_b],
            "class": [output_writer_class_a, output_writer_class_b],
        },
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    test_file = get_output_filepath("ow_func_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    test_file = get_output_filepath("ow_func_b.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    test_file = get_output_filepath("ow_class_a.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    test_file = get_output_filepath("ow_class_b.20.0.txt")
    assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_*.txt")
    cleanup_files("ow_class_*.txt")


# New tests for old style output writers


def test_save_first_last_and_multiple_times(clock_08, almost_default_grid):
    """Test save_first_timestep, save_last_timestep, and saving at multiple
    timesteps."""
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
            "function": [output_writer_function_a, output_writer_function_b],
            "class": [output_writer_class_a, output_writer_class_b],
        },
        output_interval=6.0,
        output_dir=_TEST_OUTPUT_DIR,
        save_first_timestep=True,
        save_last_timestep=True,
    )
    model.run()

    for t in ["0.0", "6.0", "12.0", "18.0", "20.0"]:
        # assert things were done correctly
        filename_bases = [
            f"ow_func_a.{t}.txt",
            f"ow_func_b.{t}.txt",
            f"ow_class_a.{t}.txt",
            f"ow_class_b.{t}.txt",
        ]
        for filename_base in filename_bases:
            truth_file = os.path.join(_TEST_DATA_DIR, f"truth_{filename_base}")
            test_file = os.path.join(os.curdir, "output", filename_base)
            assert filecmp(test_file, truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files(get_output_filepath("ow_func_*.txt"))
    cleanup_files(get_output_filepath("ow_class_*.txt"))
