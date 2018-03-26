#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
closed_node_baselevel_handler.py: controls elevation for all closed nodes.
"""
import numpy as np
from scipy.interpolate import interp1d

class ClosedNodeBaselevelHandler():
    """ClosedNodeBaselevelHandler controls elevation for closed nodes.

    The outlet lowering rate is
    specified either as a constant or through a time, elevation change textfile.

    Parameters
    ----------
    grid
    modify_boundary_nodes
    outlet_lowering_rate
    outlet_lowering_file_path
    modern_outlet_elevation


     """

    def __init__(self,
                 grid,
                 modify_boundary_nodes = False,
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
        if self.modify_boundary_nodes:
            nodes_to_lower = self.grid.status_at_node != 0
            prefactor = -1.0
        else:
            nodes_to_lower = self.grid.status_at_node == 0
            prefactor = 1.0

        # next, lower the correct nodes the desired amount

        # first, if we do not have an outlet elevation object
        if self.outlet_elevation_obj is None:

            # if this is not a watershed, we are raising the core nodes
            self.z[nodes_to_lower] += prefactor * self.outlet_lowering_rate * dt

            # if bedrock_elevation exists as a field, lower it also
            self.grid.at_node['bedrock__elevation'][nodes_to_lower] += prefactor * self.outlet_lowering_rate * dt

        # if there is an outlet elevation object
        else:
            # if bedrock_elevation exists as a field, lower it also
            # calcuate the topographic change required to match the current time's value for
            # outlet elevation. This must be done in case bedrock elevation exists, and must
            # be done before the topography is lowered

            topo_change = prefactor * (self.z[nodes_to_lower] - self.outlet_elevation_obj(self.model_time))

            if 'bedrock__elevation' in self.grid.at_node.keys():

                self.grid.at_node['bedrock__elevation'][nodes_to_lower] += topo_change

            # lower topography
            self.z[nodes_to_lower] += topo_change
