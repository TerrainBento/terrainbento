# -*- coding: utf-8 -*-
"""
model_010_basicHy.py: calculates water erosion using the
hybrid alluvium model.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         Space

@author: Charlie Shobe
@author: Katherine Barnhart
5 April 2017
"""
import numpy as np
from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                ErosionDeposition, LinearDiffuser)
class BasicHy(_ErosionModel):
    """
    A BasicHy model computes erosion of sediment and bedrock
    using dual mass conservation on the bed and in the water column. It
    applies exponential entrainment rules to account for bed cover.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the HybridAlluviumModel."""

        # Call ErosionModel's init
        super(BasicHy, self).__init__(input_file=input_file,
                                      params=params,
                                      BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters and convert units if necessary:
        K_sp = self.get_parameter_from_exponent('K_sp', raise_error=False)
        K_ss = self.get_parameter_from_exponent('K_ss', raise_error=False)

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError('A parameter for both K_rock_sp and K_rock_ss has been'
                             'provided. Only one of these may be provided')
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (self._length_factor**(1./3.))*K_ss # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError('A value for K_rock_sp or K_rock_ss  must be provided.')

        # Unit conversion for linear_diffusivity, with units L^2/T
        linear_diffusivity = ((self._length_factor ** 2.)
            * self.get_parameter_from_exponent('linear_diffusivity'))

        # Normalized settling velocity (dimensionless)
        v_sc = self.get_parameter_from_exponent('v_sc')

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        #make area_field and/or discharge_field depending on discharge_method
#        area_field = self.grid.at_node['drainage_area']
#        discharge_field = None

        # Handle solver option
        try:
            solver = self.params['solver']
        except:
            solver = 'original'

        # Instantiate a Space component
        self.eroder = ErosionDeposition(self.grid,
                            K=self.K,
                            phi=self.params['phi'],
                            F_f=self.params['F_f'],
                            v_s=v_sc,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            method='simple_stream_power',
                            discharge_method='drainage_area',
                            area_field='drainage_area',
                            solver=solver)

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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            self.eroder.K = (self.K
                             * self.pc.get_erodibility_adjustment_factor(self.model_time))
        self.eroder.run_one_step(dt,
                                 flooded_nodes=flooded,
                                 dynamic_dt=True,
                                 flow_director=self.flow_router.flow_director)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # calculate model time
        self.model_time += dt

        # Lower outlet
        self.update_outlet(dt)

        # Check walltime
        self.check_walltime()


def main():
    """Execute model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    ha = BasicHy(input_file=infile)
    ha.run()


if __name__ == '__main__':
    main()
