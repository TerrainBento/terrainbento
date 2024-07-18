# !/usr/env/python
import glob
import os

import xarray as xr

from terrainbento import Basic

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# _TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def test_write_output_raster(tmpdir, basic_raster_inputs_yaml):
    with tmpdir.as_cwd():
        basic_raster_inputs_yaml += "".join(
            [
                "\n",
                "    output_prefix: tb_raster_output\n",
                f"    output_dir: {tmpdir}\n",
            ]
        )
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")
        model.run()
        name_pattern = model.output_prefix + "*.nc"
        fs = glob.glob(os.path.join(model.output_dir, name_pattern))

        assert len(fs) == 5

        ds = xr.open_dataset(fs[0])
        ds.close()

        # todo assess raster output.

        model.remove_output_netcdfs()


def test_write_output_hex(tmpdir, basic_inputs_yaml):
    with tmpdir.as_cwd():
        basic_inputs_yaml += "".join(
            [
                "\n",
                "    output_prefix: tb_hex_output\n",
                f"    output_dir: {tmpdir}\n",
            ]
        )
        with open("params.yaml", "w") as fp:
            fp.write(basic_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        model.run()
        name_pattern = model.output_prefix + "*.nc"
        fs = glob.glob(os.path.join(model.output_dir, name_pattern))

        assert len(fs) == 5
        # ds = xr.open_dataset(fs[0])

        # todo assess hex output

        model.remove_output_netcdfs()


def test_write_synthesis_netcdf(tmpdir, basic_raster_inputs_for_nc_yaml):
    truth = os.path.join(_TEST_DATA_DIR, "truth.nc")
    with tmpdir.as_cwd():
        basic_raster_inputs_for_nc_yaml += "".join(
            [
                "\n",
                "    output_prefix: tb_synth_output\n",
                f"    output_dir: {tmpdir}\n",
            ]
        )
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_for_nc_yaml)
        model = Basic.from_file("./params.yaml")

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

        model.remove_output_netcdfs()


def test_write_synthesis_netcdf_one_field(tmpdir, basic_raster_inputs_yaml):
    with tmpdir.as_cwd():
        basic_raster_inputs_yaml += "".join(
            [
                "\n",
                "    output_prefix: tb_synth_output_one_field\n",
                f"    output_dir: {tmpdir}\n",
            ]
        )
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        truth = os.path.join(_TEST_DATA_DIR, "truth_one_field.nc")

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

        model.remove_output_netcdfs()


def test_write_synthesis_netcdf_one_field_first_timestep_false(
    tmpdir, basic_raster_inputs_yaml
):
    with tmpdir.as_cwd():
        basic_raster_inputs_yaml += "".join(
            [
                "\n",
                "    output_prefix: tb_synth_output_one_field_first_ts\n",
                f"    output_dir: {tmpdir}\n",
                "    save_first_timestep: False",
            ]
        )
        with open("params.yaml", "w") as fp:
            fp.write(basic_raster_inputs_yaml)
        model = Basic.from_file("./params.yaml")

        truth = os.path.join(_TEST_DATA_DIR, "truth_one_field_first_ts.nc")
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

        model.remove_output_netcdfs()
