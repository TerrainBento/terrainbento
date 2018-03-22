#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
single_node_baselevel_handler.py: controls elevation for a single open
boundary node.

Created on Wed Nov 15 10:36:07 2017

@author: gtucker
"""

import numpy as np
from scipy.interpolate import interp1d


class SingleNodeBaselevelHandler():
    """SingleNodeBaselevelHandler controls elevation for a single open
    boundary node, referred to here as the *outlet*."""

    def __init__(self,
                 grid,
                 outlet_lowering_rate = 0.0,
                 outlet_lowering_file_path = None,
                 modern_outlet_elevation = None,
                 **kwargs):

        # Read and remember baselevel control param, if present
        try:
            starting_outlet_elevation = self.z[self.outlet_node]

            elev_change_df = np.loadtxt(outlet_lowering_file_path, skiprows=1, delimiter =',')
            time = elev_change_df[:, 0]
            elev_change = elev_change_df[:, 1]

            scaling_factor = (np.abs(starting_outlet_elevation
                                     - modern_outlet_elevation)
                              / np.abs(elev_change[0] - elev_change[-1]))

            outlet_elevation = ((scaling_factor * elev_change_df[:, 1])
                                + starting_outlet_elevation)

            self.outlet_elevation_obj = interp1d(time, outlet_elevation)

        except KeyError:

            self.outlet_elevation_obj = None
