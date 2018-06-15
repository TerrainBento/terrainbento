# -*- coding: utf-8 -*-
"""
model_002_basicTh.py: erosion model using linear diffusion, stream
power with a smoothed threshold, and discharge proportional to drainage area.

Model 002 BasicTh

Landlab components used: LinearDiffuser, StreamPowerSmoothThresholdEroder

"""

import sys
import numpy as np

from landlab.components import LinearDiffuser, StreamPowerSmoothThresholdEroder
from terrainbento.base_class import ErosionModel


class BasicTh(ErosionModel):
    """
    A BasicTh computes erosion using linear diffusion, stream
    power with a smoothed threshold, and Q~A.
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
        """Initialize the LinDifSPThresholdModel."""

        # Call ErosionModel's init
        super(BasicTh, self).__init__(input_file=input_file,
                                      params=params,
                                      BoundaryHandlers=BoundaryHandlers,
                                        OutputWriters=OutputWriters)

        # Get Parameters and convert units if necessary:
        K_sp = self.get_parameter_from_exponent('water_erodability', raise_error=False)
        K_ss = self.get_parameter_from_exponent('water_erodability~shear_stress', raise_error=False)
        regolith_transport_parameter = (self._length_factor**2.)*self.get_parameter_from_exponent('regolith_transport_parameter') # has units length^2/time

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        threshold = self._length_factor*self.get_parameter_from_exponent('erosion__threshold') # has units length/time

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

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(self.grid,
                                                       K_sp=self.K,
                                                       m_sp=self.params['m_sp'],
                                                       n_sp=self.params['n_sp'],
                                                       threshold_sp=threshold)

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = regolith_transport_parameter)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Route flow
        self.flow_accumulator.run_one_step()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(self.flow_accumulator.depression_finder.flood_status==3)[0]

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if 'PrecipChanger' in self.boundary_handler:
            self.eroder.K = (self.K
                             * self.boundary_handler['PrecipChanger'].get_erodibility_adjustment_factor())
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

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

    ldsp = BasicTh(input_file=infile)
    ldsp.run()


if __name__ == '__main__':
    main()
