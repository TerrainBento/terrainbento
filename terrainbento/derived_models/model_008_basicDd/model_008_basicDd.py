# -*- coding: utf-8 -*-
"""
model_008_basicDd.py: erosion model using linear diffusion, stream
power with a smoothed threshold that varies with incision depth, and discharge
proportional to drainage area.

Model 008 BasicDd

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, LinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                StreamPowerSmoothThresholdEroder,
                                LinearDiffuser)
import numpy as np


class BasicDd(_ErosionModel):
    """
    A BasicDd computes erosion using linear diffusion, stream
    power with a smoothed threshold that is proportional to
    depth, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicDd."""

        # Call ErosionModel's init
        super(BasicDd, self).__init__(input_file=input_file,
                                      params=params,
                                      BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters and convert units if necessary:
        K_sp = self.get_parameter_from_exponent('K_sp', raise_error=False)
        K_ss = self.get_parameter_from_exponent('K_ss', raise_error=False)
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        self.threshold_value = self._length_factor*self.get_parameter_from_exponent('erosion__threshold') # has units length/time

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError('A parameter for both K_sp and K_ss has been'
                             'provided. Only one of these may be provided')
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (self._length_factor**(1./3.))*K_ss # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError('A value for K_sp or K_ss  must be provided.')

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros('node', 'erosion__threshold')
        self.threshold[:] = self.threshold_value

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(self.grid,
                                                       m_sp=self.params['m_sp'],
                                                       n_sp=self.params['n_sp'],
                                                       K_sp=self.K,
                                                       threshold_sp=self.threshold)

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)

    def update_erosion_threshold_values(self):
        """Updates the erosion threshold at each node based on cumulative
        erosion so far."""

        # Set the erosion threshold.
        #
        # Note that a minus sign is used because cum ero depth is negative for
        # erosion, positive for deposition.
        # The second line handles the case where there is growth, in which case
        # we want the threshold to stay at its initial value rather than
        # getting smaller.
        cum_ero = self.grid.at_node['cumulative_erosion__depth']
        cum_ero[:] = (self.z
                      - self.grid.at_node['initial_topographic__elevation'])
        self.threshold[:] = (self.threshold_value
                             - (self.thresh_change_per_depth
                                * cum_ero))
        self.threshold[self.threshold < self.threshold_value] = \
            self.threshold_value

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Calculate the new threshold values given cumulative erosion
        self.update_erosion_threshold_values()

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            self.eroder.K = (self.K
                             * self.pc.get_erodibility_adjustment_factor(self.model_time))
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

    ldsp = BasicDd(input_file=infile)
    ldsp.run()


if __name__ == '__main__':
    main()
