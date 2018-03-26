#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SingleNodeBaselevelHandler ontrols elevation for a single open boundary node.
"""

import numpy as np
from scipy.interpolate import interp1d


class SingleNodeBaselevelHandler():
    """SingleNodeBaselevelHandler controls elevation for a single open
    boundary node, referred to here as the *outlet*. The outlet lowering rate is
    specified either as a constant or through a time, elevation change textfile.

    Parameters
    ----------
    grid
    outlet_node
    outlet_lowering_rate
    outlet_lowering_file_path
    modern_outlet_elevation


     """

    def __init__(self,
                 grid,
                 outlet_node,
                 outlet_lowering_rate = 0.0,
                 outlet_lowering_file_path = None,
                 model_end_elevation = None,
                 **kwargs):

        self.grid = grid
    
        self.outlet_lowering_rate = self.params.get('outlet_lowering_rate',  0.0)
    
        try:
            file_name = self.params['outlet_lowering_file_path']
    
            model_end_elevation = self.params['model_end_elevation']
    
            model_start_elevation = self.z[self.outlet_node]
    
            elev_change_df = np.loadtxt(file_name, skiprows=1, delimiter =',')
            time = elev_change_df[:, 0]
            elev_change = elev_change_df[:, 1]
            scaling_factor = np.abs(model_start_elevation-model_end_elevation)/np.abs(elev_change[0]-elev_change[-1])
            outlet_elevation = (scaling_factor*elev_change_df[:, 1]) + model_start_elevation
    
            self.outlet_elevation_obj = interp1d(time, outlet_elevation)
    
        except KeyError:
            self.outlet_elevation_obj = None

    def run_one_step(self, dt):
        """

        """

        # determine which nodes to lower
        nodes_to_lower = self.outlet_node

        # next, lower the correct nodes the desired amount

        # first, if we do not have an outlet elevation object
        if self.outlet_elevation_obj is None:

            # calculate lowering amount and subtract
            self.z[nodes_to_lower] -= self.outlet_lowering_rate * dt

            # if bedrock_elevation exists as a field, lower it also
            self.grid.at_node['bedrock__elevation'][nodes_to_lower] -= self.outlet_lowering_rate * dt

        # if there is an outlet elevation object
        else:
            # if bedrock_elevation exists as a field, lower it also
            # calcuate the topographic change required to match the current time's value for
            # outlet elevation. This must be done in case bedrock elevation exists, and must
            # be done before the topography is lowered
            if 'bedrock__elevation' in self.grid.at_node.keys():
                topo_change = self.z[nodes_to_lower] - self.outlet_elevation_obj(self.model_time)
                self.grid.at_node['bedrock__elevation'][nodes_to_lower] -= topo_change

            # lower topography
            self.z[nodes_to_lower] = self.outlet_elevation_obj(self.model_time)
