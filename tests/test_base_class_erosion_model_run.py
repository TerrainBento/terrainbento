# coding: utf8
# !/usr/env/python

from terrainbento import Basic


def test_run_for(tmpdir, basic_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_yaml)
        model = Basic.from_file("./params.yaml")
    model._out_file_name = "run_for_output"
    model.run_for(10., 100.)
    assert model.model_time == 100.


def test_finalize(tmpdir, basic_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_yaml)
        model = Basic.from_file("./params.yaml")
    model.finalize()


def test_run(tmpdir, basic_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_yaml)
        model = Basic.from_file("./params.yaml")
    model._out_file_name = "run_output"
    model.run()
    assert model.model_time == 200.
    model.remove_output_netcdfs()
