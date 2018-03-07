# -*- coding: utf-8 -*-
"""
model_840_basicChRt.py: erosion model using cubic diffusion, basic stream
power with spatially varying K and two bedrock units, and discharge
proportional to drainage area.

Model 840 BasicChRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, CubicNonLinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                FastscapeEroder, TaylorNonLinearDiffuser)
import numpy as np


class BasicChRt(_ErosionModel):
    """
    A BasicChRt model computes erosion using cubic diffusion, basic stream
    power with two rock units, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicChRt model."""

        # Call ErosionModel's init
        super(BasicChRt, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        contact_zone__width = (self._length_factor)*self.params['contact_zone__width'] # has units length
        self.K_rock_sp = self.get_parameter_from_exponent('K_rock_sp')
        self.K_till_sp = self.get_parameter_from_exponent('K_till_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity')

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 self.K_rock_sp,
                                 self.K_till_sp,
                                 contact_zone__width)

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'],
                                      K_sp=self.erody)

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(self.grid,
                                               linear_diffusivity=linear_diffusivity,
                                               slope_crit=self.params['slope_crit'],
                                               nterms=7)

    def setup_rock_and_till(self, file_name, rock_erody, till_erody,
                            contact_width):
        """Set up lithology handling for two layers with different erodibility.

        Parameters
        ----------
        file_name : string
            Name of arc-ascii format file containing elevation of contact
            position at each grid node (or NODATA)

        Read elevation of rock-till contact from an esri-ascii format file
        containing the basal elevation value at each node, create a field for
        erodibility.

        Some considerations here:
            1. We could represent the contact between two layers either as a
               depth below present land surface, or as an altitude. Using a
               depth would allow for vertical motion, because for a fixed
               surface, the depth remains constant while the altitude changes.
               But the depth must be updated every time the surface is eroded
               or aggrades. Using an altitude avoids having to update the
               contact position every time the surface erodes or aggrades, but
               any tectonic motion would need to be applied to the contact
               position as well. Here we'll use the altitude approach because
               this model was originally written for an application with lots
               of erosion expected but no tectonics.
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

        # Create array for erodibility weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # Read the erodibility value of rock and till
        self.rock_erody = rock_erody
        self.till_erody = till_erody

        # Read and remember the contact zone characteristic width
        self.contact_width = contact_width

    def update_erodibility_field(self):
        """Update erodibility at each node based on elevation relative to
        contact elevation.

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

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Route flow
        self.flow_router.run_one_step()

        # Get IDs of flooded nodes, if any
        flooded = np.where(self.flow_router.depression_finder.flood_status==3)[0]

        # Update the erodibility field
        self.update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded,
                                 K_if_used=self.erody)

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

    chrt = BasicChRt(input_file=infile)
    chrt.run()


if __name__ == '__main__':
    main()
