import os

import numpy as np
import glob

#from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import xarray as xr

from terrainbento import Basic

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_write_output_raster():
    fp = os.path.join(_TEST_DATA_DIR, 'basic_raster_inputs.txt')
    model = Basic(input_file=fp)
    model._out_file_name ='tb_raster_output'
    model.run()
    fs = glob.glob(model._out_file_name + '*.nc')

    assert len(fs) == 5

    ds = xr.open_dataset(fs[0])

    # todo assess raster output.

    for f in fs:
        os.remove(f)


def test_write_output_hex():
    fp = os.path.join(_TEST_DATA_DIR, 'basic_inputs.txt')
    model = Basic(input_file=fp)
    model._out_file_name ='tb_hex_output'
    model.run()
    fs = glob.glob(model._out_file_name + '*.nc')

    assert len(fs) == 5
    #ds = xr.open_dataset(fs[0])

    # todo assess hex output

    for f in fs:
        os.remove(f)
