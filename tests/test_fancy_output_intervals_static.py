# coding: utf8
# !/usr/env/python

# Much of the set up to these tests borrowed from 
# test_base_class_erosion_model_output_writers.py

import pytest

import glob
import os

import numpy as np

from terrainbento import Basic, NotCoreNodeBaselevelHandler
from terrainbento.utilities import filecmp
from terrainbento.output_writers import StaticIntervalOutputWriter

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def empty_model(clock_08):
    class EmptyModel:
        def __init__(self):
            self.clock = clock_08

    return EmptyModel()

'''
def output_writer_function_a(model):
    average_elevation = np.mean(model.z[model.grid.core_nodes])
    with open("ow_func_a." + str(model.model_time) + ".txt", "w") as f:
        f.write(str(average_elevation))


def output_writer_function_b(model):
    minimum_elevation = np.min(model.z[model.grid.core_nodes])
    with open("ow_func_b." + str(model.model_time) + ".txt", "w") as f:
        f.write(str(minimum_elevation))

class output_writer_class_a(object):
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        average_change = np.mean(self.change[self.model.grid.core_nodes])
        with open(
            "ow_class_a." + str(self.model.model_time) + ".txt", "w"
        ) as f:
            f.write(str(average_change))


class output_writer_class_b(object):
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        min_change = np.min(self.change[self.model.grid.core_nodes])
        with open(
            "ow_class_b." + str(self.model.model_time) + ".txt", "w"
        ) as f:
            f.write(str(min_change))

class output_writer_class_static(StaticIntervalOutputWriter):
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        min_change = np.min(self.change[self.model.grid.core_nodes])
        with open(
            "ow_class_b." + str(self.model.model_time) + ".txt", "w"
        ) as f:
            f.write(str(min_change))
            
def cleanup_files(searchpath):
    files = glob.glob(searchpath)
    for f in files:
        os.remove(f)

'''


@pytest.mark.parametrize("in_name, add_id, out_name", [
    (None, True, "static-interval-output-writer"),
    (None, False, "static-interval-output-writer"),
    ('given_nameT', True, "given_nameT"),
    ('given_nameF', False, "given_nameF"),
    ])
def test_names(empty_model, in_name, add_id, out_name):
    # Make a few writer to check if the id value is handled correctly
    for i in range(3):
        if in_name is None:
            writer = StaticIntervalOutputWriter(
                    empty_model,
                    add_id=add_id,
                    )
        else:
            writer = StaticIntervalOutputWriter(
                    empty_model,
                    name=in_name,
                    add_id=add_id,
                    )
        if add_id:
            assert writer.name == out_name + f"-id{writer.id}"
        else:
            assert writer.name == out_name
            
    

@pytest.mark.parametrize("intervals, times, error_type", [
    ([1,2,3], [1,2,3], AssertionError), # Both args defined
    ('a', None, NotImplementedError), # Bad arg type
    (None, 'a', NotImplementedError), # Bad arg type
    (['a'], None, AssertionError), # Bad arg type in list
    (None, ['a'], AssertionError), # Bad arg type in list
    ])
def test_interval_times_bad_input(empty_model, intervals, times, error_type):
    with pytest.raises(error_type):
        writer = StaticIntervalOutputWriter(
                empty_model,
                intervals=intervals, 
                times=times, 
                )

@pytest.mark.parametrize("intervals, times, output_times", [
    (None, None, [20.0]), # stop time for clock_08
    (5,   None, [5.0, 10.0, 15.0, 20.0]), # Test single interval duration (int)
    (5.0, None, [5.0, 10.0, 15.0, 20.0]), # Test single interval duration (float)
    (None, 5,   [5.0]), # Test single output time (int)
    (None, 5.0, [5.0]), # Test single output time (float)
    ([1,2,3], None, [1.0, 3.0, 6.0]), # Test list of integer intervals
    ([1.0,2.0,3.0], None, [1.0, 3.0, 6.0]), # Test list of float intervals
    (None, [1,2,3], [1.0, 2.0, 3.0]), # Test list of integer times
    (None, [1.0,2.0,3.0], [1.0, 2.0, 3.0]), # Test list of float times
    ])
def test_intervals_times_correct_input(empty_model, intervals, times, output_times):
    writer = StaticIntervalOutputWriter(
            empty_model,
            intervals=intervals, 
            times=times, 
            )
    # assert that output_times is an iterator
    # Check for the __next__ function?
    # I want to make sure I can iterate through output_times, not that 
    # output_times is iterable (e.g. see if iter(obj) fails)
    assert hasattr(writer.times_iter, '__next__')

    for correct_out in output_times:
        writer_out = next(writer.times_iter)
        assert writer_out == correct_out and \
               type(writer_out) == type(correct_out)

def test_intervals_repeat(empty_model):
    intervals = [1, 2, 3]
    output_times = [1.0, 3.0, 6.0, 7.0, 9.0, 12.0, 13.0, 15.0, 18.0, 19.0]
    # Note that clock_08 stop time is 20.0

    writer = StaticIntervalOutputWriter(
            empty_model,
            intervals=intervals, 
            intervals_repeat=True,
            times=None, 
            )

    assert hasattr(writer.times_iter, '__next__')

    for correct_out in output_times:
        writer_out = next(writer.times_iter)
        assert writer_out == correct_out and \
               type(writer_out) == type(correct_out)
    
    # Note that the writer.times_iter will be an infinite iterator.


''' old tests for reference

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
    assert filecmp("ow_func_a.20.0.txt", truth_file) is True

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
    assert filecmp("ow_class_a.20.0.txt", truth_file) is True

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
    assert filecmp("ow_func_a.20.0.txt", truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    assert filecmp("ow_func_b.20.0.txt", truth_file) is True

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
        output_writers={
            "class": [output_writer_class_a, output_writer_class_b]
        },
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    assert filecmp("ow_class_a.20.0.txt", truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    assert filecmp("ow_class_b.20.0.txt", truth_file) is True

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
    assert filecmp("ow_func_a.20.0.txt", truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    assert filecmp("ow_func_b.20.0.txt", truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    assert filecmp("ow_class_a.20.0.txt", truth_file) is True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    assert filecmp("ow_class_b.20.0.txt", truth_file) is True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_*.txt")
    cleanup_files("ow_class_*.txt")
'''
