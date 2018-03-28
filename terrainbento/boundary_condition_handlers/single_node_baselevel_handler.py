#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SingleNodeBaselevelHandler controls elevation for a single open boundary node.
"""
import os
import numpy as np
from scipy.interpolate import interp1d


class SingleNodeBaselevelHandler():
    """Control the elevation of a single open boundary node.

    The SingleNodeBaselevelHandler controls the elevation of a single open
    boundary node, referred to here as the *outlet*. The outlet lowering rate is
    specified either as a constant or through a time or through a textfile that
    specifies the elevation change through time.

    Methods
    -------
    run_one_step(dt)
    """

    def __init__(self,
                 grid,
                 outlet_node,
                 outlet_lowering_rate = None,
                 outlet_lowering_file_path = None,
                 model_end_elevation = None,
                 **kwargs):
        """
        Parameters
        ----------
        grid : landlab model grid
        outlet_node : int
            Node ID of the outlet node.
        outlet_lowering_rate : float, optional
            Lowering rate of the outlet node. One of `outlet_lowering_rate` and
            `outlet_lowering_file_path` is required. Units are implied by the
            model grids spatial scale and the time units of `dt`. Negative
            values mean that the outlet lowers.
        outlet_lowering_file_path : str, optional
            Lowering lowering history file path. One of `outlet_lowering_rate`
            and `outlet_lowering_file_path` is required. Units are implied by
            the model grids spatial scale and the time units of `dt`.
            This file should be readable with
            `np.loadtxt(filename, skiprows=1, delimiter=',')`
            Its first column is time and its second colum is the elevation
            change at the outlet since the onset of the model run. Negative
            values mean the outlet lowers.
        model_end_elevation : float, optional
            Elevation of the outlet at the end of the model run duration. When
            the outlet is lowered based on an outlet_lowering_file_path, a
            `model_end_elevation` can be set such that lowering is scaled
            based on the starting and ending outlet elevation. Default behavior
            is to not scale the lowering pattern.

        Examples
        --------
        Start by creating a landlab model grid.

        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid(5, 5)
        >>> z = mg.add_zeros('node', 'topographic__elevation')
        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        Now import the `SingleNodeBaselevelHandler` and instantiate.

        >>> from terrainbento.boundary_condition_handlers import SingleNodeBaselevelHandler
        >>> bh = SingleNodeBaselevelHandler(mg,
        ...                                 outlet_node = 0,
        ...                                 outlet_lowering_rate = -0.1)
        >>> bh.run_one_step(10.0)

        We should expect that node 0 has lowered by one, to an elevation of -1.

        >>> print(z.reshape(mg.shape))
        [[-1.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        More complex baselevel histories can be provided with a
        `outlet_lowering_file_path`.

        """
        self.model_time = 0.0
        self.grid = grid
        self.outlet_node = outlet_node
        self.outlet_lowering_rate = outlet_lowering_rate
        self.z = self.grid.at_node['topographic__elevation']

        if (outlet_lowering_file_path is None) and (outlet_lowering_rate is None):
            raise ValueError(('SingleNodeBaselevelHandler requires one of '
                              'outlet_lowering_rate and outlet_lowering_file_path'))
        else:
            if (outlet_lowering_rate is None):
                # initialize outlet elevation object
                if os.path.exists(outlet_lowering_file_path):

                    model_start_elevation = self.z[self.outlet_node]
                    elev_change_df = np.loadtxt(outlet_lowering_file_path, skiprows=1, delimiter =',')
                    time = elev_change_df[:, 0]
                    elev_change = elev_change_df[:, 1]

                    if model_end_elevation is None:
                        scaling_factor = 1.0
                    else:
                        scaling_factor = np.abs(model_start_elevation-model_end_elevation)/np.abs(elev_change[0]-elev_change[-1])
                    outlet_elevation = (scaling_factor*elev_change_df[:, 1]) + model_start_elevation
                    self.outlet_elevation_obj = interp1d(time, outlet_elevation)
                    self.outlet_lowering_rate = None
                else:
                    raise ValueError(('The outlet_lowering_file_path provided '
                                      'to SingleNodeBaselevelHandler does not '
                                      'exist.'))
            elif (outlet_lowering_file_path is None):
                self.outlet_lowering_rate = outlet_lowering_rate
                self.outlet_elevation_obj = None
            else:
                raise ValueError(('Both an outlet_lowering_rate and a '
                                  'outlet_lowering_file_path have been provided '
                                  'to SingleNodeBaselevelHandler. Please provide '
                                  'only one.'))

    def run_one_step(self, dt):
        """
        Run SingleNodeBaselevelHandler forward and update outlet node elevation.

        Parameters
        ----------
        dt : float
            Duration of model time to advance forward.
        """
        # increment model time
        self.model_time += dt

        # first, if we do not have an outlet elevation object
        if self.outlet_elevation_obj is None:

            # calculate lowering amount and subtract
            self.z[self.outlet_node] += self.outlet_lowering_rate * dt

            # if bedrock_elevation exists as a field, lower it also
            if 'bedrock__elevation' in self.grid.at_node:
                self.grid.at_node['bedrock__elevation'][self.outlet_node] += self.outlet_lowering_rate * dt

        # if there is an outlet elevation object
        else:
            # if bedrock_elevation exists as a field, lower it also
            # calcuate the topographic change required to match the current time's value for
            # outlet elevation. This must be done in case bedrock elevation exists, and must
            # be done before the topography is lowered
            if 'bedrock__elevation' in self.grid.at_node:
                topo_change = self.z[self.outlet_node] - self.outlet_elevation_obj(self.model_time)
                self.grid.at_node['bedrock__elevation'][self.outlet_node] -= topo_change

            # lower topography
            self.z[self.outlet_node] = self.outlet_elevation_obj(self.model_time)
