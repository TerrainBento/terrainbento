#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:59:39 2018

@author: barnhark
"""

import sys
import os

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal
from nose.tools import assert_raises

from terrainbento.boundary_condition_handlers import CaptureNodeBaselevelHandler
from landlab import RasterModelGrid, HexModelGrid


def text_hex():
    "Test using a hex grid"

    mg = HexModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')
    
    bh = CaptureNodeBaselevelHandler(mg, capture_node = 3, 
                                     capture_incision_rate = -3.0, 
                                     capture_start_time = 10, 
                                     capture_stop_time = 20, 
                                     post_capture_incision_rate = -0.1)
    bh.run_one_step(10)
    
    
def test_no_stop_time():
    """Test with no stop time"""

    mg = RasterModelGrid(5, 5)
    z = mg.add_zeros('node', 'topographic__elevation')
    
    bh = CaptureNodeBaselevelHandler(mg, capture_node = 3, 
                                     capture_incision_rate = -3.0, 
                                     capture_start_time = 0)
    
    for i in range(10):
        bh.run_one_step(10)
    
    
    assert z[3] == -3.0*10*10