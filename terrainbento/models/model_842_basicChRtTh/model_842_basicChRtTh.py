# -*- coding: utf-8 -*-
"""
model_840_basicChRtTh.py: erosion model using cubic diffusion, basic stream
power with a threshold and spatially varying K and two bedrock units, and discharge
proportional to drainage area.

Model 842 BasicChRtTh

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, CubicNonLinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                StreamPowerSmoothThresholdEroder, TaylorNonLinearDiffuser)
import numpy as np


class BasicChRtTh(_ErosionModel):
    """
    A BasicChRt model computes erosion using cubic diffusion, basic stream
    power with two rock units, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicChRt model."""

        # Call ErosionModel's init
        super(BasicChRtTh, self).__init__(input_file=input_file,
                                          params=params,
                                          BaselevelHandlerClass=BaselevelHandlerClass)

        contact_zone__width = (self._length_factor)*self.params['contact_zone__width'] # has units length
        self.K_rock_sp = self.get_parameter_from_exponent('K_rock_sp')
        self.K_till_sp = self.get_parameter_from_exponent('K_till_sp')
        rock_erosion__threshold = self.get_parameter_from_exponent('rock_erosion__threshold')
        till_erosion__threshold = self.get_parameter_from_exponent('till_erosion__threshold')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity')

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 self.K_rock_sp,
                                 self.K_till_sp,
                                 rock_erosion__threshold,
                                 till_erosion__threshold,
                                 contact_zone__width)

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a StreamPowerSmoothThresholdEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(self.grid,
                                                       K_sp=self.erody,
                                                       threshold_sp=self.threshold,
                                                       m_sp=self.params['m_sp'],
                                                       n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(self.grid,
                                               linear_diffusivity=linear_diffusivity,
                                               slope_crit=self.params['slope_crit'],
                                               nterms=7)

    def setup_rock_and_till(self, file_name, rock_erody, till_erody,
                            rock_thresh, till_thresh, contact_width):
        """Set up lithology handling for two layers with different erodibility.

        Parameters
        ----------
        file_name : string
            Name of arc-ascii format file containing elevation of contact
            position at each grid node (or NODATA)
        rock_erody : float
            Water erosion coefficient for bedrock
        till_erody : float
            Water erosion coefficient for till
        rock_thresh : float
            Water erosion threshold for bedrock
        till_thresh : float
            Water erosion threshold for till
        contact_width : float [L]
            Characteristic width of the interface zone between rock and till

        Read elevation of rock-till contact from an esri-ascii format file
        containing the basal elevation value at each node, create a field for
        erodibility.
        """
        from landlab.io import read_esri_ascii

        # Read input data on rock-till contact elevation
        read_esri_ascii(file_name, grid=self.grid,
                        name='rock_till_contact__elevation',
                        halo=1)

        # Get a reference to the rock-till field
        self.rock_till_contact = self.grid.at_node['rock_till_contact__elevation']

        # Create field for erodibility
        if 'substrate__erodibility' in self.grid.at_node:
            self.erody = self.grid.at_node['substrate__erodibility']
        else:
            self.erody = self.grid.add_zeros('node', 'substrate__erodibility')

        # Create field for threshold values
        if 'erosion__threshold' in self.grid.at_node:
            self.threshold = self.grid.at_node['erosion__threshold']
        else:
            self.threshold = self.grid.add_zeros('node', 'erosion__threshold')

        # Create array for erodibility weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # Read the erodibility value of rock and till
        self.rock_erody = rock_erody
        self.till_erody = till_erody

        # Read the threshold values for rock and till
        self.rock_thresh = rock_thresh
        self.till_thresh = till_thresh

        # Read and remember the contact zone characteristic width
        self.contact_width = contact_width

    def update_erodibility_and_threshold_fields(self):
        """Update erodibility and threshold at each node based on elevation
        relative to contact elevation.

        To promote smoothness in the solution, the erodibility at a given point
        (x,y) is set as follows:

            1. Take the difference between elevation, z(x,y), and contact
               elevation, b(x,y): D(x,y) = z(x,y) - b(x,y). This number could
               be positive (if land surface is above the contact), negative
               (if we're well within the rock), or zero (meaning the rock-till
               contact is right at the surface).
            2. Define a smoothing function as:
                $F(D) = 1 / (1 + exp(-D/D*))$
               This sigmoidal function has the property that F(0) = 0.5,
               F(D >> D*) = 1, and F(-D << -D*) = 0.
                   Here, D* describes the characteristic width of the "contact
               zone", where the effective erodibility is a mixture of the two.
               If the surface is well above this contact zone, then F = 1. If
               it's well below the contact zone, then F = 0.
            3. Set the erodibility using F:
                $K = F K_till + (1-F) K_rock$
               So, as F => 1, K => K_till, and as F => 0, K => K_rock. In
               between, we have a weighted average.
            4. Threshold values are set similarly.

        Translating these symbols into variable names:

            z = self.elev
            b = self.rock_till_contact
            D* = self.contact_width
            F = self.erody_wt
            K_till = self.till_erody
            K_rock = self.rock_erody
        """

        # Update the erodibility weighting function (this is "F")
        D_over_D_star = ((self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                                         / self.contact_width)

        # truncate D_over_D star to remove potential for overflow in exponent
        D_over_D_star[D_over_D_star < -100.0] = -100.0
        D_over_D_star[D_over_D_star > 100.0] = 100.0

        self.erody_wt[self.data_nodes] = (1.0 / (1.0 + np.exp(-D_over_D_star)))

        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            erode_factor = self.pc.get_erodibility_adjustment_factor(self.model_time)
            self.till_erody = self.K_till_sp * erode_factor
            self.rock_erody = self.K_rock_sp * erode_factor

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (self.erody_wt * self.till_erody
                         + (1.0 - self.erody_wt) * self.rock_erody)

        # Calculate the effective thresholds using weighted averaging
        self.threshold[:] = (self.erody_wt * self.till_thresh
                             + (1.0 - self.erody_wt) * self.rock_thresh)

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Update the erodibility and threshold field
        self.update_erodibility_and_threshold_fields()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt,
                                   dynamic_dt=True,
                                   if_unstable='raise',
                                   courant_factor=0.1)

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

    chrt = BasicChRtTh(input_file=infile)
    chrt.run()


if __name__ == '__main__':
    main()
