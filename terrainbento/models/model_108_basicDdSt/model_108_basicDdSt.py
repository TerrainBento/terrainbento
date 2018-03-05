# -*- coding: utf-8 -*-
"""
model_108_basicDdSt.py: erosion model with stochastic
rainfall, and water erosion proportional to stream power in excess of a
threshold that increases progressively with incision depth.

Model 108 BasicDdSt

The hydrology aspect models discharge and erosion across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) Hortonian infiltration model. Includes stream-power
erosion plus linear diffusion.

The hydrology uses calculation of drainage area using the standard "D8"
approach (assuming the input grid is a raster; "DN" if not), then modifies it
by running a lake-filling component. It then iterates through a sequence of
storm and interstorm periods. Storm depth is drawn at random from a gamma
distribution, and storm duration from an exponential distribution; storm
intensity is then depth divided by duration. Given a storm precipitation
intensity $P$, the runoff production rate $R$ [L/T] is calculated using:

$R = P - I (1 - \exp ( -P / I ))$

where $I$ is the soil infiltration capacity. At the sub-grid scale, soil
infiltration capacity is assumed to have an exponential distribution of which
$I$ is the mean. Hence, there are always some spots within any given grid cell
that will generate runoff. This approach yields a smooth transition from
near-zero runoff (when $I>>P$) to $R \approx P$ (when $P>>I$), without a
"hard threshold."

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution, LinearDiffuser, StreamPowerSmoothThresholdEroder

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.stochastic_erosion_model import _StochasticErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                LinearDiffuser, StreamPowerSmoothThresholdEroder)

import numpy as np


class BasicDdSt(_StochasticErosionModel):
    """
    A BasicDdSt computes erosion using (1) unit
    stream power with a threshold, (2) linear nhillslope diffusion, and
    (3) generation of a random sequence of runoff events across a topographic
    surface.

    Examples
    --------
    >>> from erosion_model import StochasticRainDepthDepThresholdModel
    >>> my_pars = {}
    >>> my_pars['dt'] = 1.0
    >>> my_pars['run_duration'] = 1.0
    >>> my_pars['infiltration_capacity'] = 1.0
    >>> my_pars['K_sp'] = 1.0
    >>> my_pars['threshold_sp'] = 1.0
    >>> my_pars['linear_diffusivity'] = 0.01
    >>> my_pars['mean_storm_duration'] = 0.002
    >>> my_pars['mean_interstorm_duration'] = 0.008
    >>> my_pars['mean_storm_depth'] = 0.025
    >>> srt = StochasticRainDepthDepThresholdModel(params=my_pars)
    Warning: no DEM specified; creating 4x5 raster grid
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicDdSt."""

        # Call ErosionModel's init
        super(BasicDdSt, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters:
        K_sp = self.get_parameter_from_exponent('K_stochastic_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        self.threshold_value = self._length_factor*self.get_parameter_from_exponent('erosion__threshold') # has units length/time

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']

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

        # Get the infiltration-capacity parameter
        infiltration_capacity = (self._length_factor)*self.params['infiltration_capacity']# has units length per time
        self.infilt = infiltration_capacity

        # Keep a reference to drainage area
        self.area = self.grid.at_node['drainage_area']

        # Run flow routing and lake filler
        self.flow_router.run_one_step()

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros('node', 'erosion__threshold')
        self.threshold[:] = self.threshold_value

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(self.grid,
                                                       m_sp=self.params['m_sp'],
                                                       n_sp=self.params['n_sp'],
                                                       K_sp=K_sp,
                                                       use_Q=self.discharge,
                                                       threshold_sp=self.threshold)

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)

    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""
        if self.rain_rate > 0.0 and self.infilt > 0.0:
            runoff = self.rain_rate - (self.infilt *
                                       (1.0 -
                                        np.exp(-self.rain_rate / self.infilt)))
            if runoff < 0:
                runoff = 0
        else:
            runoff = self.rain_rate
        self.discharge[:] = runoff * self.area
        return runoff

    def update_threshold_field(self):
        """Update the threshold based on cumulative erosion depth."""
        cum_ero = self.grid.at_node['cumulative_erosion__depth']
        cum_ero[:] = (self.z
                      - self.grid.at_node['initial_topographic__elevation'])
        self.threshold[:] = (self.threshold_value
                             - (self.thresh_change_per_depth * cum_ero))
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

        # Handle water erosion
        self.handle_water_erosion_with_threshold(dt, flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # calculate model time
        self.model_time += dt

        # Lower outlet
        self.update_outlet(dt)

        # Check walltime
        self.check_walltime()

    def handle_water_erosion_with_threshold(self, dt, flooded):
        """Handle water erosion.

        This function takes the place of the _BaseSt function of the name
        handle_water_erosion_with_threshold in order to handle water erosion
        correctly for model BasicDdSt.
        """
        # (if we're varying precipitation parameters through time, update them)
        if self.opt_var_precip:
            self.intermittency_factor, self.mean_storm__intensity = self.pc.get_current_precip_params(self.model_time)

        # If we're handling duration deterministically, as a set fraction of
        # time step duration, calculate a rainfall intensity. Otherwise,
        # assume it's already been calculated.
        if not self.opt_stochastic_duration:
            self.rain_rate = np.random.exponential(self.mean_storm__intensity)
            dt_water = dt * self.intermittency_factor
        else:
            dt_water = dt

        # Calculate discharge field
        area = self.grid.at_node['drainage_area']
        if self.rain_rate > 0.0 and self.infilt > 0.0:
            runoff = self.rain_rate - (self.infilt *
                                       (1.0 -
                                        np.exp(-self.rain_rate / self.infilt)))
        else:
            runoff = self.rain_rate

        self.discharge[:] = runoff * area

        # Handle water erosion:
        #
        #   If we are running stochastic duration, then self.rain_rate will
        #   have been calculated already. It might be zero, in which case we
        #   are between storms, so we don't do water erosion.
        #
        #   If we're NOT doing stochastic duration, then we'll run water
        #   erosion for one or more sub-time steps, each with its own
        #   randomly drawn precipitation intensity.
        #
        if self.opt_stochastic_duration and self.rain_rate > 0.0:
            self.update_threshold_field()
            runoff = self.calc_runoff_and_discharge()
            self.eroder.run_one_step(dt, flooded_nodes=flooded)
        elif not self.opt_stochastic_duration:
            dt_water = ((dt * self.intermittency_factor)
                         / float(self.n_sub_steps))
            for i in range(self.n_sub_steps):
                self.rain_rate = \
                    self.rain_generator.generate_from_stretched_exponential(
                        self.scale_factor, self.shape_factor)
                self.update_threshold_field()
                runoff = self.calc_runoff_and_discharge()
                self.eroder.run_one_step(dt_water, flooded_nodes=flooded)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    em = BasicDdSt(input_file=infile)
    em.run()


if __name__ == '__main__':
    main()
