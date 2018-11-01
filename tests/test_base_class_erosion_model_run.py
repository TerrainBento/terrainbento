# coding: utf8
#! /usr/env/python

import os

from terrainbento import Basic

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_run_for():
    fp = os.path.join(_TEST_DATA_DIR, "basic_inputs.yaml")
    model = Basic(input_file=fp)
    model._out_file_name = "run_for_output"
    model.run_for(10., 100.)
    assert model.model_time == 100.


def test_finalize():
    fp = os.path.join(_TEST_DATA_DIR, "basic_inputs.yaml")
    model = Basic(input_file=fp)
    model.finalize()


def test_run():
    fp = os.path.join(_TEST_DATA_DIR, "basic_inputs.yaml")
    model = Basic(input_file=fp)
    model._out_file_name = "run_output"
    model.run()
    assert model.model_time == 200.
    model.remove_output_netcdfs()
