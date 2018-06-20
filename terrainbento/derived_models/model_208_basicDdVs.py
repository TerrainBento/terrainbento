#! /usr/env/python
"""
model_208_basicDdVs.py: erosion model using linear diffusion,
thresholded stream power, and discharge proportional to effective drainage
area. The threshold varies in space and time, increasing linearly with
cumulative incision depth below initial topographic surface.

Model 208 BasicDdVs

"vsa" stands for "variable source area".

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerEroder, LinearDiffuser

"""

import sys
import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicDdVs(ErosionModel):
    """
    A BasicDdVs computes erosion using linear diffusion,
    "smoothly thresholded" stream power in which the threshold increases with
    cumulative erosion depth, and Q ~ A exp( -b S / A).

    "VSA" stands for "variable source area".
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
        """Initialize the VSADepthDepThresholdModel."""

        # Call ErosionModel's init
        super(BasicDdVs, self).__init__(input_file=input_file,
                                        params=params,
                                        BoundaryHandlers=BoundaryHandlers,
                                        OutputWriters=OutputWriters)

        self.K_sp = self.get_parameter_from_exponent('water_erodability')
        regolith_transport_parameter = (self._length_factor**2.)*self.get_parameter_from_exponent('regolith_transport_parameter') # has units length^2/time

        recharge_rate = (self._length_factor)*self.params['recharge_rate'] # has units length per time
        soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        K_hydraulic_conductivity = (self._length_factor)*self.params['K_hydraulic_conductivity'] # has units length per time

        self.threshold_value = self._length_factor*self.get_parameter_from_exponent('erosion__threshold') # has units length/time

        # Add a field for effective drainage area
        if 'effective_drainage_area' in self.grid.at_node:
            self.eff_area = self.grid.at_node['effective_drainage_area']
        else:
            self.eff_area = self.grid.add_zeros('node',
                                                'effective_drainage_area')

        # Get the effective-area parameter
        self.sat_param = (K_hydraulic_conductivity*soil_thickness*self.grid.dx)/(recharge_rate)

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros('node', 'erosion__threshold')
        self.threshold[:] = self.threshold_value

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(self.grid,
                                                use_Q=self.eff_area,
                                                       K_sp=self.K_sp,
                                                       m_sp=self.params['m_sp'],
                                                       n_sp=self.params['n_sp'],
                                                       threshold_sp=self.threshold)

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params['thresh_change_per_depth']

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = regolith_transport_parameter)



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
        self.flow_accumulator.run_one_step()

        # Update effective runoff ratio
        self.calc_effective_drainage_area()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(self.flow_accumulator.depression_finder.flood_status==3)[0]

        # Zero out effective area in flooded nodes
        self.eff_area[flooded] = 0.0

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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if 'PrecipChanger' in self.boundary_handler:
            self.eroder.K = (self.K_sp
                             * self.boundary_handler['PrecipChanger'].get_erodibility_adjustment_factor())
        self.eroder.run_one_step(dt)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    my_model = BasicDdVs(input_file=infile)
    my_model.run()


if __name__ == '__main__':
    main()
