# -*- coding: utf-8 -*-
"""
model_410_basicHySa.py: erosion model using depth-dependent linear diffusion,
hybrid alluvium river erosion, and discharge proportional to drainage
area.

Model 410 BasicHySa

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         Space, DepthDependentDiffuser,
                         ExponentialWeatherer

@author: Charlie Shobe
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                Space, DepthDependentDiffuser,
                                ExponentialWeatherer)
import numpy as np


class BasicHySa(_ErosionModel):
    """
    A BasicHySa computes erosion using linear diffusion, hybrid alluvium,
    and Q~A.

    It creates soil through weathering and fluvial bedrock erosion,
    and consideres soil thickness in calculating hillslope diffusion.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicSa."""

        # Call ErosionModel's init
        super(BasicHySa, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        self.K_br = self.get_parameter_from_exponent('K_rock_sp')
        self.K_sed = self.get_parameter_from_exponent('K_sed_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time
        v_sc = self.get_parameter_from_exponent('v_sc') # normalized settling velocity. Unitless.

        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor)*self.params['soil_transport_decay_depth']  # has units length
        max_soil_production_rate = (self._length_factor)*self.params['max_soil_production_rate'] # has units length per time
        soil_production_decay_depth = (self._length_factor)*self.params['soil_production_decay_depth']   # has units length

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        #set methods and fields. K's and sp_crits need to be field names
        method = 'simple_stream_power'
        discharge_method = 'discharge_field'
        area_field = None
        discharge_field = 'surface_water__discharge'

        # Instantiate a SPACE component
        self.eroder = Space(self.grid,
                            K_sed=self.K_sed,
                            K_br=self.K_br,
                            F_f=self.params['F_f'],
                            phi=self.params['phi'],
                            H_star=self.params['H_star'],
                            v_s=v_sc,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            method=method,
                            discharge_method=discharge_method,
                            area_field=area_field,
                            discharge_field=discharge_field,
                            solver=self.params['solver'])

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

        # Set soil thickness and bedrock elevation
        try:
            initial_soil_thickness = self.params['initial_soil_thickness']
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(self.grid,
                                               linear_diffusivity=linear_diffusivity,
                                               soil_transport_decay_depth=soil_transport_decay_depth)

        self.weatherer = ExponentialWeatherer(self.grid,
                                              max_soil_production_rate=max_soil_production_rate,
                                              soil_production_decay_depth=soil_production_decay_depth)

        self.grid.at_node['soil__depth'][:] = \
            self.grid.at_node['topographic__elevation'] - \
            self.grid.at_node['bedrock__elevation']

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]
        #print('There are ' + str(len(flooded)) + ' flooded nodes')

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            erode_factor = self.pc.get_erodibility_adjustment_factor(self.model_time)
            self.eroder.K_sed = self.K_sed * erode_factor
            self.eroder.K_br = self.K_br * erode_factor

        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node['bedrock__elevation']
        b[:] = np.minimum(b, self.grid.at_node['topographic__elevation'])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
        self.diffuser.run_one_step(dt)

        # calculate model time
        self.model_time += dt

        # Lower outlet
        self.update_outlet(dt)

        # Check walltime
        self.check_walltime()

        # Check stability
        self.check_stability()

    def check_stability(self):
        """Check stability and exit if unstable."""
        fields = self.grid.at_node.keys()
        for f in fields:
            if (np.any(np.isnan(self.grid.at_node[f])) or
                np.any(np.isinf(self.grid.at_node[f]))):

                # model is unstable, write message and exit.
                with open('model_failed.txt', 'w') as f:
                    f.write('This model run became unstable\n')

                exit

def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    hysa = BasicHySa(input_file=infile)
    hysa.run()


if __name__ == '__main__':
    main()
