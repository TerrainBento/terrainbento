# coding: utf8
# !/usr/env/python

# This file has tests for the new style output writers.

# Missing tests:
# - test save_first_timestep and save_last_timestep for new style
# - test longer sequence of outputs
# - New style in general
# - Provide bad input new-style formats?
#
# - UserWarning for model step and output writer divisibility (in 
#   ErosionModel._update_output_times
# - Try to break model time passing next output time? (assertion in 
#   ErosionModel.write_output()
#

import glob
import os

import numpy as np

from terrainbento import Basic, NotCoreNodeBaselevelHandler
from terrainbento.utilities import filecmp

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


def cleanup_files(searchpath):
    files = glob.glob(searchpath)
    for f in files:
        os.remove(f)


