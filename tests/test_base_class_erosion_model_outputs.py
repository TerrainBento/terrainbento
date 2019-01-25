# coding: utf8
# !/usr/env/python
import glob
import os

import xarray as xr

from terrainbento import Basic

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_write_output_raster(tmpdir, basic_raster_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")
        model._out_file_name = "tb_raster_output"
        model.run()
        fs = glob.glob(model._out_file_name + "*.nc")

        assert len(fs) == 5

        ds = xr.open_dataset(fs[0])
        ds.close()

        # todo assess raster output.


def test_write_output_hex(tmpdir, basic_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        model._out_file_name = "tb_hex_output"
        model.run()
        fs = glob.glob(model._out_file_name + "*.nc")

        assert len(fs) == 5
        # ds = xr.open_dataset(fs[0])

        # todo assess hex output


def test_write_synthesis_netcdf(tmpdir, basic_raster_inputs_for_nc_yaml):
    truth = os.path.join(_TEST_DATA_DIR, "truth.nc")
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_for_nc_yaml)
        model = Basic.from_file("./params.yaml")

        model._out_file_name = "tb_synth_output"
        model.run()

        ds = model.to_xarray_dataset(time_unit="years", space_unit="meter")

        out_fn = "tb_output.nc"
        model.save_to_xarray_dataset(
            filename=out_fn, time_unit="years", space_unit="meter"
        )

        output = xr.open_dataset(out_fn, decode_times=False)
        truth = xr.open_dataset(truth, decode_times=False)

        assert truth.dims == output.dims
        assert truth.dims == ds.dims

        assert truth.equals(output) is True
        assert truth.equals(ds) is True

        output.close()
        truth.close()
        ds.close()


def test_write_synthesis_netcdf_one_field(tmpdir, basic_raster_inputs_yaml):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        truth = os.path.join(_TEST_DATA_DIR, "truth_one_field.nc")

        model._out_file_name = "tb_synth_output_one_field"
        model.run()

        ds = model.to_xarray_dataset(time_unit="years", space_unit="meter")

        out_fn = "tb_output_one_field.nc"
        model.save_to_xarray_dataset(
            filename=out_fn, time_unit="years", space_unit="meter"
        )

        output = xr.open_dataset(out_fn, decode_times=False)
        truth = xr.open_dataset(truth, decode_times=False)

        assert truth.dims == output.dims
        assert truth.dims == ds.dims

        assert truth.equals(output) is True
        assert truth.equals(ds) is True

        output.close()
        truth.close()
        ds.close()


def test_write_synthesis_netcdf_one_field_first_timestep_false(
    tmpdir, basic_raster_inputs_yaml
):
    with tmpdir.as_cwd():
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        truth = os.path.join(_TEST_DATA_DIR, "truth_one_field_first_ts.nc")
        model.save_first_timestep = False
        model._out_file_name = "tb_synth_output_one_field_first_ts"
        model.run()

        ds = model.to_xarray_dataset(time_unit="years", space_unit="meter")

        out_fn = "tb_output_one_field_first_ts.nc"
        model.save_to_xarray_dataset(
            filename=out_fn, time_unit="years", space_unit="meter"
        )

        output = xr.open_dataset(out_fn, decode_times=False)
        truth = xr.open_dataset(truth, decode_times=False)

        assert truth.dims == output.dims
        assert truth.dims == ds.dims

        assert truth.equals(output) is True
        assert truth.equals(ds) is True

        output.close()
        truth.close()
        ds.close()
