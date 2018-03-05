# -*- coding: utf-8 -*-
"""
model_300_basicStVs.py: models discharge and erosion across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) a variable-source-area (VSA) runoff-generation model.

Model 300 BasicStVs

This model combines linear diffusion and basic stream power with stochastic
variable source area (VSA) hydrology. It inherits from the ErosionModel 
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster; "DN" if not), then modifies it by running a
lake-filling component. It then iterates through a sequence of storm and
interstorm periods. Storm depth is drawn at random from a gamma distribution,
and storm duration from an exponential distribution; storm intensity is then
depth divided by duration. Given a storm precipitation intensity $P$, the
discharge $Q$ [L$^3$/T] is calculated using:

$Q = PA - T\lambda S [1 - \exp (-PA/T\lambda S) ]$

where $T$ is the soil transmissivity and $\lambda$ is cell width.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution, StreamPowerEroder, LinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.stochastic_erosion_model import _StochasticErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                StreamPowerEroder, LinearDiffuser)

import numpy as np


class BasicStVs(_StochasticErosionModel):
    """
    A BasicStVs generates a random sequency of
    runoff events across a topographic surface, calculating the resulting
    water discharge at each node.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the StochasticDischargeHortonianModel."""

        # Call ErosionModel's init
        super(BasicStVs,
              self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)
        # Get Parameters:
        K_sp = self.get_parameter_from_exponent('K_stochastic_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time

        soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        K_hydraulic_conductivity = (self._length_factor)*self.params['K_hydraulic_conductivity'] # has units length per time

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        if 'surface_water__discharge' not in self.grid.at_node:
            self.grid.add_zeros('node', 'surface_water__discharge')
        self.discharge = self.grid.at_node['surface_water__discharge']   

        # Add a field for subsurface discharge                                 
        if 'subsurface_water__discharge' not in self.grid.at_node:
            self.grid.add_zeros('node', 'subsurface_water__discharge')
        self.qss = self.grid.at_node['subsurface_water__discharge']  

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.trans = (K_hydraulic_conductivity*soil_thickness)
        assert (self.trans > 0.0), 'Transmissivity must be > 0'
        self.tlam = self.trans * self.grid._dx  # assumes raster

        # Run flow routing and lake filler
        self.flow_router.run_one_step()

        # Keep a reference to drainage area and steepest-descent slope
        self.area = self.grid.at_node['drainage_area']
        self.slope = self.grid.at_node['topographic__steepest_slope']

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(self.grid,
                                        use_Q=self.discharge,
                                        K_sp=K_sp,
                                        m_sp=self.params['m_sp'],
                                        n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)


    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""

        # Here's the total (surface + subsurface) discharge
        pa = self.rain_rate * self.area

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = self.tlam * self.slope[np.where(self.slope > 0.0)[0]]

        # Subsurface discharge: zero where slope is flat
        self.qss[np.where(self.slope <= 0.0)[0]] = 0.0
        self.qss[np.where(self.slope > 0.0)[0]] = (tls 
                    * (1.0 - np.exp(-pa[np.where(self.slope > 0.0)[0]] / tls)))

        # Surface discharge = total minus subsurface
        #
        # Note that roundoff errors can sometimes produce a tiny negative
        # value when qss and pa are close; make sure these are set to 0
        self.discharge[:] = pa - self.qss
        self.discharge[self.discharge < 0.0] = 0.0


    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        
        # Route flow
        self.flow_router.run_one_step()
        
        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Handle water erosion
        self.handle_water_erosion(dt, flooded)

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

    dm = BasicStVs(input_file=infile)
    dm.run()


if __name__ == '__main__':
    main()
