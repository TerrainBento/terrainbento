# -*- coding: utf-8 -*-
"""
model_600_basicVsSa.py: erosion model using depth-dependent linear diffusion,
basic stream power, and discharge proportional to effective drainage area, with
a transmissivity parameter that depends on space-time varying soil thickness.

Model 600 BasicVsSa

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerEroder, DepthDependentDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

import sys
import numpy as np

from landlab.components import (StreamPowerEroder, DepthDependentDiffuser,
                                ExponentialWeatherer)
from terrainbento.base_class import ErosionModel


class BasicSaVs(ErosionModel):
    """
    A BasicSaVs computes erosion using depth-dependent linear diffusion, basic
    stream power, and Q ~ A exp( -c H S / A); H = soil thickness.

    This "c" parameter has dimensions of length, and is defined as
    c = K dx / R, where K is saturated hydraulic conductivity, dx is grid
    spacing, and R is recharge.
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None):
        """Initialize the BasicVsSa."""

        # Call ErosionModel's init
        super(BasicVsSa, self).__init__(input_file=input_file,
                                        params=params,
                                        BoundaryHandlers=BoundaryHandlers)

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent('K_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor)*self.params['soil_transport_decay_depth']  # has units length
        max_soil_production_rate = (self._length_factor)*self.params['max_soil_production_rate'] # has units length per time
        soil_production_decay_depth = (self._length_factor)*self.params['soil_production_decay_depth']   # has units length

        recharge_rate = (self._length_factor)*self.params['recharge_rate'] # has units length per time
        K_hydraulic_conductivity = (self._length_factor)*self.params['K_hydraulic_conductivity'] # has units length per time

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

        # Add a field for effective drainage area
        if 'effective_drainage_area' in self.grid.at_node:
            self.eff_area = self.grid.at_node['effective_drainage_area']
        else:
            self.eff_area = self.grid.add_zeros('node',
                                                'effective_drainage_area')

        # Get the effective-length parameter
        self.sat_len = (K_hydraulic_conductivity*self.grid.dx)/(recharge_rate)

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(self.grid,
                                        use_Q=self.eff_area,
                                        K_sp=self.K_sp,
                                        m_sp=self.params['m_sp'],
                                        n_sp=self.params['n_sp'])

        # Instantiate a DepthDependentDiffuser component
        self.diffuser = DepthDependentDiffuser(self.grid,
                                               linear_diffusivity=linear_diffusivity,
                                               soil_transport_decay_depth=soil_transport_decay_depth)

        self.weatherer = ExponentialWeatherer(self.grid,
                                              max_soil_production_rate=max_soil_production_rate,
                                              soil_production_decay_depth=soil_production_decay_depth)



    def calc_effective_drainage_area(self):
        """Calculate and store effective drainage area.

        Effective drainage area is defined as:

        $A_{eff} = A \exp ( c H S / A) = A R_r$

        where $H$ is soil thickness, $S$ is downslope-positive steepest
        gradient, $A$ is drainage area, $R_r$ is the runoff ratio, and $c$ is
        the saturation length parameter.
        """

        area = self.grid.at_node['drainage_area']
        slope = self.grid.at_node['topographic__steepest_slope']
        soil = self.grid.at_node['soil__depth']
        cores = self.grid.core_nodes
        self.eff_area[cores] = (area[cores] * (np.exp(-self.sat_len
                                                      * soil[cores]
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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if 'PrecipChanger' in self.boundary_handler:
            self.eroder.K = (self.K_sp
                             * self.boundary_handler['PrecipChanger'].get_erodibility_adjustment_factor())
        self.eroder.run_one_step(dt)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node['bedrock__elevation']
        b[:] = np.minimum(b, self.grid.at_node['topographic__elevation'])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

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

    vssa = BasicVsSa(input_file=infile)
    vssa.run()


if __name__ == '__main__':
    main()
