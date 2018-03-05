# -*- coding: utf-8 -*-
"""
model_000_basic.py: erosion model using linear diffusion, basic stream
power, and discharge proportional to drainage area.

Model 000 Basic

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, LinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                FastscapeEroder, LinearDiffuser)

import numpy as np
from scipy.interpolate import interp1d

class BasicCv(_ErosionModel):
    """
    A BasicCV computes erosion using linear diffusion, basic stream
    power, and Q~A.

    It also has basic climate change
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicCv model."""
        # Call ErosionModel's init
        super(BasicCv, self).__init__(input_file=input_file,
                                      params=params,
                                      BaselevelHandlerClass=BaselevelHandlerClass)


        K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity')


        self.climate_factor = self.params['climate_factor']
        self.climate_constant_date = self.params['climate_constant_date']

        time = [0, self.climate_constant_date, self.params['run_duration']]
        K = [K_sp*self.climate_factor, K_sp, K_sp]
        self.K_through_time = interp1d(time, K)

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=K[0],
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Update erosion based on climate
        self.eroder.K = float(self.K_through_time(self.model_time))

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

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

    ldsp = BasicCv(input_file=infile)
    ldsp.run()


if __name__ == '__main__':
    main()
