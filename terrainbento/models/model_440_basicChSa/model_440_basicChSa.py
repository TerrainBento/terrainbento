# -*- coding: utf-8 -*-
"""
model_440_basicChSa.py: erosion model using depth-dependent cubic diffusion
with a soil layer, basic stream power, and discharge proportional to drainage
area.

Model 440 BasicChSa

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, DepthDependentCubicDiffuser,
                         ExponentialWeatherer

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                FastscapeEroder, DepthDependentTaylorDiffuser,
                                ExponentialWeatherer)
import numpy as np


class BasicChSa(_ErosionModel):
    """
    A BasicChSa model computes erosion using depth-dependent cubic diffusion
    with a soil layer, basic stream power, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicChSa model."""

        # Call ErosionModel's init
        super(BasicChSa, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        self.K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor)*self.params['soil_transport_decay_depth']  # has units length
        max_soil_production_rate = (self._length_factor)*self.params['max_soil_production_rate'] # has units length per time
        soil_production_decay_depth = (self._length_factor)*self.params['soil_production_decay_depth']   # has units length

        # Create soil thickness (a.k.a. depth) field
        if 'soil__depth' in self.grid.at_node:
            soil_thickness = self.grid.at_node['soil__depth']
        else:
            soil_thickness = self.grid.add_zeros('node', 'soil__depth')

        # Create bedrock elevation field
        if 'bedrock__elevation' in self.grid.at_node:
            bedrock_elev = self.grid.at_node['bedrock__elevation']
        else:
            bedrock_elev = self.grid.add_zeros('node', 'bedrock__elevation')

        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=self.K_sp,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

        # Instantiate a weathering component
        self.weatherer = ExponentialWeatherer(self.grid,
                                              max_soil_production_rate=max_soil_production_rate,
                                              soil_production_decay_depth=soil_production_decay_depth)

        # Instantiate a soil-transport component
        self.diffuser = DepthDependentTaylorDiffuser(self.grid,
                                                    linear_diffusivity=linear_diffusivity,
                                                    slope_crit=self.params['slope_crit'],
                                                    soil_transport_decay_depth=soil_transport_decay_depth,
                                                    nterms=11)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            self.eroder.K = (self.K_sp
                             * self.pc.get_erodibility_adjustment_factor(self.model_time))

        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node['bedrock__elevation']
        b[:] = np.minimum(b, self.grid.at_node['topographic__elevation'])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Do some soil creep
        self.diffuser.run_one_step(dt,
                                   dynamic_dt=True,
                                   if_unstable='raise',
                                   courant_factor=0.1)

        # calculate model time
        self.model_time += dt

        # Lower outlet
        self.update_outlet(dt)

        # Check walltime
        self.check_walltime()

def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    cdsp = BasicChSa(input_file=infile)
    cdsp.run()


if __name__ == '__main__':
    main()
