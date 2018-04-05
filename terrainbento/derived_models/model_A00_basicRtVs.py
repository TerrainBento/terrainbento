# -*- coding: utf-8 -*-
"""
model_A00_basicVsRt.py: erosion model using linear diffusion, basic stream
power with 2 lithologies (rock and till), and discharge proportional to
effective drainage area.

Model A00 BasicVsRt

Landlab components used: FlowRouter, StreamPowerEroder, LinearDiffuser
"""

import sys
import numpy as np

from landlab.components import StreamPowerEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicRtVs(ErosionModel):
    """
    A BasicVsRt computes erosion using linear diffusion, basic stream
    power with 2 lithologies, and Q ~ A exp( -b S / A).
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None):
        """Initialize the BasicVsRt."""

        # Call ErosionModel's init
        super(BasicVsRt, self).__init__(input_file=input_file,
                                        params=params,
                                        BoundaryHandlers=BoundaryHandlers)
        contact_zone__width = (self._length_factor)*self.params['contact_zone__width'] # has units length
        self.K_rock_sp = self.get_parameter_from_exponent('K_rock_sp')
        self.K_till_sp = self.get_parameter_from_exponent('K_till_sp')
        linear_diffusivity = (self._length_factor**2.)*self.get_parameter_from_exponent('linear_diffusivity')

        recharge_rate = (self._length_factor)*self.params['recharge_rate'] # has units length per time
        soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        K_hydraulic_conductivity = (self._length_factor)*self.params['K_hydraulic_conductivity'] # has units length per time

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 self.K_rock_sp,
                                 self.K_till_sp,
                                 contact_zone__width)

        # Add a field for effective drainage area
        if 'effective_drainage_area' in self.grid.at_node:
            self.eff_area = self.grid.at_node['effective_drainage_area']
        else:
            self.eff_area = self.grid.add_zeros('node',
                                                'effective_drainage_area')

        # Get the effective-area parameter
        self.sat_param = (K_hydraulic_conductivity*soil_thickness*self.grid.dx)/(recharge_rate)

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(self.grid,
                                        K_sp=self.erody,
                                        m_sp=self.params['m_sp'],
                                        n_sp=self.params['n_sp'],
                                        use_Q=self.eff_area)

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
        core = self.grid.core_nodes
        if self.contact_width > 0.0:
            self.erody_wt[core] = (
                1.0 / (1.0 + np.exp(-(self.z[core]
                                      - self.rock_till_contact[core])
                                     / self.contact_width)))
        else:
            self.erody_wt[core] = 0.0
            self.erody_wt[np.where(self.z > self.rock_till_contact)[0]] = 1.0

        # (if we're varying K through time, update that first)
        if 'PrecipChanger' in self.boundary_handler:
            erode_factor = self.boundary_handler['PrecipChanger'].get_erodibility_adjustment_factor()
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

        # Update effective runoff ratio
        self.calc_effective_drainage_area()

        # Zero out effective area in flooded nodes
        self.eff_area[self.flow_router.depression_finder.flood_status==3] = 0.0

        # Update the erodibility field
        self.update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # calculate model time
        self.model_time += dt

        # Update boundary conditions
        self.update_boundary_conditions(dt)

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

    vsrt = BasicVsRt(input_file=infile)
    vsrt.run()


if __name__ == '__main__':
    main()
