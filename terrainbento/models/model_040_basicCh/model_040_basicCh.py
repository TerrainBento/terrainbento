# -*- coding: utf-8 -*-
"""
model_040_basicCh.py: erosion model using cubic diffusion, basic stream
power, and discharge proportional to drainage area.

Model 040 BasicCh

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, LinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                FastscapeEroder, TaylorNonLinearDiffuser)
import numpy as np


class BasicCh(_ErosionModel):
    """
    A BasicCh computes erosion using cubic diffusion, basic stream
    power, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicCh."""

        # Call ErosionModel's init
        super(BasicCh, self).__init__(input_file=input_file,
                                      params=params,
                                      BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=self.K_sp,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(self.grid,
                                               linear_diffusivity=linear_diffusivity,
                                               slope_crit=self.params['slope_crit'],
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

    cdsp = BasicCh(input_file=infile)
    cdsp.run()


if __name__ == '__main__':
    main()
