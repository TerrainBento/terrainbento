# -*- coding: utf-8 -*-
"""
model_800_basicRt.py: erosion model using linear diffusion, basic stream
power with spatially varying K and two bedrock units, and discharge
proportional to drainage area.

Model 800 BasicRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, LinearDiffuser

@author: gtucker
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                FastscapeEroder, LinearDiffuser)
import numpy as np


class BasicRt(_ErosionModel):
    """
    A BasicRt model computes erosion using linear diffusion, basic stream
    power with two rock units, and Q~A.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicRt model."""

        # Call ErosionModel's init
        super(BasicRt, self).__init__(input_file=input_file,
                                      params=params,
                                      BaselevelHandlerClass=BaselevelHandlerClass)

        # Get Parameters and convert units if necessary:
        contact_zone__width = (self._length_factor)*self.params['contact_zone__width'] # has units length
        K_rock_sp = self.get_parameter_from_exponent('K_rock_sp', raise_error=False)
        K_rock_ss = self.get_parameter_from_exponent('K_rock_ss', raise_error=False)
        K_till_sp = self.get_parameter_from_exponent('K_till_sp', raise_error=False)
        K_till_ss = self.get_parameter_from_exponent('K_till_ss', raise_error=False)
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity') # has units length^2/time

        # check that a stream power and a shear stress parameter have not both been given
        # first for rock Ks
        if K_rock_sp != None and K_rock_ss != None:
            raise ValueError('A parameter for both K_rock_sp and K_rock_ss has been'
                             'provided. Only one of these may be provided')
        elif K_rock_sp != None or K_rock_ss != None:
            if K_rock_sp != None:
                self.K_rock = K_rock_sp
            else:
                self.K_rock = (self._length_factor**(1./3.))*K_rock_ss # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError('A value for K_rock_sp or K_rock_ss  must be provided.')

        # Then for Till Ks
        if K_till_sp != None and K_till_ss != None:
            raise ValueError('A parameter for both K_till_sp and K_rock_ss has been'
                             'provided. Only one of these may be provided')
        elif K_till_sp != None or K_till_ss != None:
            if K_till_sp != None:
                self.K_till = K_till_sp
            else:
                self.K_till = (self._length_factor**(1./3.))*K_till_ss # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError('A value for K_till_sp or K_till_ss  must be provided.')

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 self.K_rock,
                                 self.K_till,
                                 contact_zone__width)

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder = DepressionFinderAndRouter)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=self.erody,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity = linear_diffusivity)

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
        read_esri_ascii(file_name, grid=self.grid, halo=1,
                        name='rock_till_contact__elevation')

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
        self.erody_wt[self.data_nodes] = (1.0
                            / (1.0
                               + np.exp(-(self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                                         / self.contact_width)))

        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            erode_factor = self.pc.get_erodibility_adjustment_factor(self.model_time)
            self.till_erody = self.K_till * erode_factor
            self.rock_erody = self.K_rock * erode_factor

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

    brt = BasicRt(input_file=infile)
    brt.run()


if __name__ == '__main__':
    main()
