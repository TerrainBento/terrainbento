# -*- coding: utf-8 -*-
"""
model_810_basicHyRt.py: erosion model using linear diffusion, the hybrid
alluvium stream erosion model, discharge proportional to drainage area, and
two lithologies: rock and till.

Model 810 BasicHyRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         Space, LinearDiffuser

IMPORTANT: This model allows changes in erodibility and threshold for bedrock
abd sediment INDEPENDENTLY, meaning that weighting functions etc. exist for
both.

@author: Charlie Shobe
@author: Katherine Barnhart
"""

from erosion_model.erosion_model import _ErosionModel
from landlab.components import (FlowAccumulator, DepressionFinderAndRouter,
                                ErosionDeposition, LinearDiffuser)
import numpy as np

class BasicHyRt(_ErosionModel):
    """
    A BasicHyRt computes erosion using linear diffusion, hybrid alluvium
    stream erosion, Q~A, and two lithologies: rock and till.
    """

    def __init__(self, input_file=None, params=None,
                 BaselevelHandlerClass=None):
        """Initialize the BasicHyRt."""

        # Call ErosionModel's init
        super(BasicHyRt, self).__init__(input_file=input_file,
                                        params=params,
                                        BaselevelHandlerClass=BaselevelHandlerClass)

        contact_zone__width = (self._length_factor
                               * self.params['contact_zone__width']) # L
        self.K_rock_sp = self.get_parameter_from_exponent('K_rock_sp')
        self.K_till_sp = self.get_parameter_from_exponent('K_till_sp')

        linear_diffusivity = ((self._length_factor ** 2)
                * self.get_parameter_from_exponent('linear_diffusivity'))

        v_sc = self.get_parameter_from_exponent('v_sc') # normalized settling velocity. Unitless.

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 rock_erody_br=self.K_rock_sp,
                                 till_erody_br=self.K_till_sp,
                                 rock_thresh_br=0.0,
                                 till_thresh_br=0.0,
                                 contact_width=contact_zone__width)

        # Instantiate a FlowAccumulator with DepressionFinderAndRouter using D8 method
        self.flow_router = FlowAccumulator(self.grid,
                                           flow_director='D8',
                                           depression_finder=DepressionFinderAndRouter)

        # Handle solver option
        try:
            solver = self.params['solver']
        except:
            solver = 'original'

        # Instantiate an ErosionDeposition ("hybrid") component
        self.eroder = ErosionDeposition(self.grid,
                            K='K_br',
                            F_f=self.params['F_f'],
                            phi=self.params['phi'],
                            v_s=v_sc,
                            m_sp=self.params['m_sp'],
                            n_sp=self.params['n_sp'],
                            method='simple_stream_power',
                            discharge_method='drainage_area',
                            area_field='drainage_area',
                            solver=solver)

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(self.grid,
                                       linear_diffusivity=linear_diffusivity)

    def setup_rock_and_till(self,
                            file_name='file',
                            rock_erody_br=1,
                            till_erody_br=1,
                            rock_thresh_br=0,
                            till_thresh_br=0,
                            contact_width=1):
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

        # Create field for rock erodability
        if 'K_br' in self.grid.at_node:
            self.erody_br = self.grid.at_node['K_br']
        else:
            self.erody_br = self.grid.add_ones('node', 'K_br')
            self.erody_br[:] = rock_erody_br

        # field for rock threshold values
        if 'sp_crit_br' in self.grid.at_node:
            self.threshold_br = self.grid.at_node['sp_crit_br']
        else:
            self.threshold_br = self.grid.add_ones('node', 'sp_crit_br')
            self.threshold_br[:] = rock_thresh_br

        # Create array for erodibility weighting function for BEDROCK
        self.erody_wt_br = np.zeros(self.grid.number_of_nodes)

        # Read the erodibility value of rock and till
        self.rock_erody_br = rock_erody_br
        self.till_erody_br = till_erody_br

        # Read the threshold values for rock and till
        self.rock_thresh_br = rock_thresh_br
        self.till_thresh_br = till_thresh_br

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
        self.erody_wt_br[self.data_nodes] = (1.0
                            / (1.0
                               + np.exp(-(self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                                         / self.contact_width)))

        # (if we're varying K through time, update that first)
        if self.opt_var_precip:
            erode_factor = self.pc.get_erodibility_adjustment_factor(self.model_time)
            self.till_erody_br = self.K_till_sp * erode_factor
            self.rock_erody_br = self.K_rock_sp * erode_factor

        # Calculate the effective BEDROCK erodibilities using weighted averaging
        self.erody_br[:] = (self.erody_wt_br * self.till_erody_br
                         + (1.0 - self.erody_wt_br) * self.rock_erody_br)

        # Calculate the effective BEDROCK thresholds using weighted averaging
        self.threshold_br[:] = (self.erody_wt_br * self.till_thresh_br
                             + (1.0 - self.erody_wt_br) * self.rock_thresh_br)

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

    thrt = BasicHyRt(input_file=infile)
    thrt.run()


if __name__ == '__main__':
    main()
