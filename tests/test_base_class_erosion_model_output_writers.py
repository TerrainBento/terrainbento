# coding: utf8
#! /usr/env/python

import os
import numpy as np
import glob

from terrainbento import Basic
from terrainbento.utilities import *


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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
        with open("ow_class_a." + str(self.model.model_time) + ".txt", "w") as f:
            f.write(str(average_change))


class output_writer_class_b(object):
    def __init__(self, model):
        self.model = model
        self.change = model.grid.at_node["cumulative_elevation_change"]

    def run_one_step(self):
        min_change = np.min(self.change[self.model.grid.core_nodes])
        with open("ow_class_b." + str(self.model.model_time) + ".txt", "w") as f:
            f.write(str(min_change))


def cleanup_files(searchpath):
    files = glob.glob(searchpath)
    for f in files:
        os.remove(f)


def test_one_function_writer():
    params = {
        "save_first_timestep": False,
        "clock": CLOCK_08,
        "node_spacing": 100.0,
        "regolith_transport_parameter": 0.0,
        "water_erodability": 0.0,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -1},
    }
    # construct and run model
    model = Basic(params=params, OutputWriters=output_writer_function_a)
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    assert filecmp("ow_func_a.20.0.txt", truth_file) == True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_a.*.txt")


def test_one_class_writer():
    params = {
        "save_first_timestep": False,
        "clock": CLOCK_08,
        "node_spacing": 100.0,
        "regolith_transport_parameter": 0.0,
        "water_erodability": 0.0,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -1},
    }
    # construct and run model
    model = Basic(params=params, OutputWriters=output_writer_class_a)
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    assert filecmp("ow_class_a.20.0.txt", truth_file) == True

    model.remove_output_netcdfs()
    cleanup_files("ow_class_a.*.txt")


def test_two_function_writers():
    params = {
        "save_first_timestep": False,
        "clock": CLOCK_08,
        "node_spacing": 100.0,
        "regolith_transport_parameter": 0.0,
        "water_erodability": 0.0,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -1},
    }
    # construct and run model
    model = Basic(
        params=params,
        OutputWriters=[output_writer_function_a, output_writer_function_b],
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    assert filecmp("ow_func_a.20.0.txt", truth_file) == True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    assert filecmp("ow_func_b.20.0.txt", truth_file) == True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_*.txt")


def test_two_class_writers():
    params = {
        "save_first_timestep": False,
        "clock": CLOCK_08,
        "node_spacing": 100.0,
        "regolith_transport_parameter": 0.0,
        "water_erodability": 0.0,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -1},
    }
    # construct and run model
    model = Basic(
        params=params, OutputWriters=[output_writer_class_a, output_writer_class_b]
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    assert filecmp("ow_class_a.20.0.txt", truth_file) == True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    assert filecmp("ow_class_b.20.0.txt", truth_file) == True

    model.remove_output_netcdfs()
    cleanup_files("ow_class_*.txt")


def test_all_four_writers():
    params = {
        "save_first_timestep": False,
        "clock": CLOCK_08,
        "node_spacing": 100.0,
        "regolith_transport_parameter": 0.0,
        "water_erodability": 0.0,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -1},
    }
    # construct and run model
    model = Basic(
        params=params,
        OutputWriters=[
            output_writer_function_a,
            output_writer_function_b,
            output_writer_class_a,
            output_writer_class_b,
        ],
    )
    model.run()

    # assert things were done correctly
    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_a.20.0.txt")
    assert filecmp("ow_func_a.20.0.txt", truth_file) == True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_func_b.20.0.txt")
    assert filecmp("ow_func_b.20.0.txt", truth_file) == True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_a.20.0.txt")
    assert filecmp("ow_class_a.20.0.txt", truth_file) == True

    truth_file = os.path.join(_TEST_DATA_DIR, "truth_ow_class_b.20.0.txt")
    assert filecmp("ow_class_b.20.0.txt", truth_file) == True

    model.remove_output_netcdfs()
    cleanup_files("ow_func_*.txt")
    cleanup_files("ow_class_*.txt")
