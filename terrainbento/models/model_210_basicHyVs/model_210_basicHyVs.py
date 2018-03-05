# -*- coding: utf-8 -*-
"""
model_210_basicHyVs.py: erosion model using linear diffusion,
hybrid alluvium, and discharge proportional to effective drainage
area.

Model 210 BasicHyVs

"vsa" stands for "variable source area".

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerEroder, LinearDiffuser

@author: Charlie Shobe
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                ErosionDeposition, LinearDiffuser)
import numpy as np


class BasicHyVs(_ErosionModel):
    """
    A BasicHyVs computes erosion using linear diffusion,
    hybrid alluvium fluvial erosion, and Q ~ A exp( -b S / A).

    "VSA" stands for "variable source area".
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicHyVs."""

        # Call ErosionModel's init
        super(BasicHyVs, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        self.K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = ((self._length_factor ** 2)
                * self.get_parameter_from_exponent('linear_diffusivity')) # has units length^2/time
        recharge_rate = (self._length_factor
                         * self.params['recharge_rate']) # L/T
        soil_thickness = (self._length_factor
                          * self.params['initial_soil_thickness']) # L
        K_hydraulic_conductivity = (self._length_factor
                                    * self.params['K_hydraulic_conductivity']) # has units length per time

        v_sc = self.get_parameter_from_exponent('v_sc') # normalized settling velocity. Unitless.

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # set methods and fields. K's and sp_crits need to be field names
        method = 'simple_stream_power'
        discharge_method = 'drainage_area'
        area_field = 'effective_drainage_area'
        discharge_field = None

        # Add a field for effective drainage area
        if 'effective_drainage_area' in self.grid.at_node:
            self.eff_area = self.grid.at_node['effective_drainage_area']
        else:
            self.eff_area = self.grid.add_zeros('node',
                                                'effective_drainage_area')

        # Get the effective-area parameter
        self.sat_param = ((K_hydraulic_conductivity * soil_thickness
                           * self.grid.dx) / recharge_rate)

        # Handle solver option
        try:
            solver = self.params['solver']
        except KeyError:
            solver = 'original'

        # Instantiate a SPACE component
        self.eroder = ErosionDeposition(self.grid,
                            K=self.K_sp,
                            F_f=self.params['F_f'],
                            phi=self.params['phi'],
                            v_s=v_sc,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            method=method,
                            discharge_method=discharge_method,
                            area_field=area_field,
                            discharge_field=discharge_field,
                            solver=solver)

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)


    def calc_effective_drainage_area(self):
        """Calculate and store effective drainage area.

        Effective drainage area is defined as:

        $A_{eff} = A \exp ( \alpha S / A) = A R_r$

        where $S$ is downslope-positive steepest gradient, $A$ is drainage
        area, $R_r$ is the runoff ratio, and $\alpha$ is the saturation
        parameter.
        """

        area = self.grid.at_node['drainage_area']
        slope = self.grid.at_node['topographic__steepest_slope']
        cores = self.grid.core_nodes
        self.eff_area[cores] = (area[cores] * (np.exp(-self.sat_param
                                                      * slope[cores]
                                                      / area[cores])))

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()

        # Update effective runoff ratio
        self.calc_effective_drainage_area()

        # Zero out effective area in flooded nodes
        self.eff_area[self.flow_router.depression_finder.flood_status==3] = 0.0

        # Do some erosion
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            self.eroder.K = (self.K_sp
                             * self.pc.get_erodibility_adjustment_factor(self.model_time))
        self.eroder.run_one_step(dt)

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

    my_model = BasicHyVs(input_file=infile)
    my_model.run()


if __name__ == '__main__':
    main()
