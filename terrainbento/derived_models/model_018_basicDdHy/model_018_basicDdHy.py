# -*- coding: utf-8 -*-
"""
model_018_basicDdHy.py: erosion model with hybrid alluvium and a threshold
that varies with cumulative incision depth.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
LinearDiffuser, and HybridAlluvium

@author: Charlie Shobe, 5 May 2017
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                LinearDiffuser, ErosionDeposition)
import numpy as np

class BasicDdHy(_ErosionModel):
    """
    A BasicDdHy computes erosion using 1) the hybrid alluvium component
    with a threshold that varies with cumulative incision depth, the linear
    diffusion component.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """
        Initialize the BasicDdHy
        """

        # Call ErosionModel's init
        super(BasicDdHy, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = ((self._length_factor ** 2)  # L2/T
                * self.get_parameter_from_exponent('linear_diffusivity'))
        v_s = self.get_parameter_from_exponent('v_sc') # unitless
        self.sp_crit = (self._length_factor  # L/T
                * self.get_parameter_from_exponent('erosion__threshold'))

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros('node', 'erosion__threshold')
        self.threshold[:] = self.sp_crit  #starting value

        # Handle solver option
        try:
            solver = self.params['solver']
        except:
            solver = 'original'

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(self.grid,
                            K=self.K_sp,
                            F_f=self.params['F_f'],
                            phi=self.params['phi'],
                            v_s=v_s,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            sp_crit='erosion__threshold',
                            method='threshold_stream_power',
                            discharge_method='drainage_area',
                            area_field='drainage_area',
                            solver=solver)

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity=linear_diffusivity)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Calculate cumulative erosion and update threshold
        cum_ero = self.grid.at_node['cumulative_erosion__depth']
        cum_ero[:] = (self.z
                     - self.grid.at_node['initial_topographic__elevation'])
        self.threshold[:] = (self.sp_crit
                             - (self.thresh_change_per_depth * cum_ero))
        self.threshold[self.threshold < self.sp_crit] = self.sp_crit

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            self.eroder.K = (self.K_sp
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

    em = BasicDdHy(input_file=infile)
    em.run()


if __name__ == '__main__':
    main()
