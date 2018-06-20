#! /usr/env/python
"""
model_C00_basicSaRt.py: erosion model using depth-dependent linear
diffusion, basic stream power with till and rocklayers, and discharge
proportional to drainage area.

Model C00 BasicSaRt

Landlab components used: FlowRouter, FastscapeStreamPower,
DepthDependentDiffuser, ExponentialWeatherer

"""
import sys
import numpy as np

from landlab.components import (FastscapeEroder, DepthDependentDiffuser,
                                ExponentialWeatherer)
from terrainbento.base_class import ErosionModel


class BasicRtSa(ErosionModel):
    """
    A BasicSaRt computes erosion using linear diffusion, basic
    stream power with rock and till layers, and Q~A.

    It creates soil through weathering and consideres soil thickness
    in calculating hillslope diffusion.
    """

    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
        """Initialize the BasicSaRt."""

        # Call ErosionModel's init
        super(BasicSaRt, self).__init__(input_file=input_file,
                                        params=params,
                                        BoundaryHandlers=BoundaryHandlers,
                                        OutputWriters=OutputWriters)
        contact_zone__width = (self._length_factor)*self.params['contact_zone__width'] # has units length
        self.K_rock_sp = self.get_parameter_from_exponent('K_rock_sp')
        self.K_till_sp = self.get_parameter_from_exponent('K_till_sp')
        regolith_transport_parameter = (self._length_factor**2.)*self.get_parameter_from_exponent('regolith_transport_parameter')

        # Set up rock-till
        self.setup_rock_and_till(self.params['rock_till_file__name'],
                                 self.K_rock_sp,
                                 self.K_till_sp,
                                 contact_zone__width)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid,
                                      K_sp=self.erody,
                                      m_sp=self.params['m_sp'],
                                      n_sp=self.params['n_sp'])

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

        # Set soil thickness and bedrock elevation
        try:
            initial_soil_thickness = (self._length_factor)*self.params['initial_soil_thickness'] # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value

        soil_transport_decay_depth = (self._length_factor)*self.params['soil_transport_decay_depth']  # has units length
        max_soil_production_rate = (self._length_factor)*self.params['max_soil_production_rate'] # has units length per time
        soil_production_decay_depth = (self._length_factor)*self.params['soil_production_decay_depth']   # has units length

        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(self.grid,
                                               linear_diffusivity=regolith_transport_parameter,
                                               soil_transport_decay_depth=soil_transport_decay_depth)

        self.weatherer = ExponentialWeatherer(self.grid,
                                              max_soil_production_rate=max_soil_production_rate,
                                              soil_production_decay_depth=soil_production_decay_depth)

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
                        name='rock_till_contact__elevation', halo=1)

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
                        + np.exp(-(self.z[self.data_nodes]
                                   - self.rock_till_contact[self.data_nodes])
                                  / self.contact_width)))

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
        self.flow_accumulator.run_one_step()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(self.flow_accumulator.depression_finder.flood_status==3)[0]

        # Update the erodibility field
        self.update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded,
                                 K_if_used=self.erody)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node['bedrock__elevation']
        b[:] = np.minimum(b, self.grid.at_node['topographic__elevation'])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
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

    sart = BasicSaRt(input_file=infile)
    sart.run()


if __name__ == '__main__':
    main()
