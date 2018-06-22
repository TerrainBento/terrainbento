#! /usr/env/python
"""
model_802_basicThRt.py: erosion model using linear diffusion, stream
power with a smoothed threshold, discharge proportional to drainage area, and
two lithologies: rock and till.

Model 802 BasicThRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, LinearDiffuser

"""

import sys
import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicRtTh(ErosionModel):
    """
    A BasicThRt computes erosion using linear diffusion, stream
    power with a smoothed threshold, Q~A, and two lithologies: rock and till.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicThRt."""

        # Call ErosionModel's init
        super(BasicThRt, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        contact_zone__width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length
        self.K_rock_sp = self.get_parameter_from_exponent("K_rock_sp")
        self.K_till_sp = self.get_parameter_from_exponent("K_till_sp")
        rock_erosion__threshold = self.get_parameter_from_exponent(
            "rock_erosion__threshold"
        )
        till_erosion__threshold = self.get_parameter_from_exponent(
            "till_erosion__threshold"
        )
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        # Set up rock-till
        self.setup_rock_and_till(
            self.params["rock_till_file__name"],
            self.K_rock_sp,
            self.K_till_sp,
            rock_erosion__threshold,
            till_erosion__threshold,
            contact_zone__width,
        )

        # Instantiate a StreamPowerSmoothThresholdEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.erody,
            threshold_sp=self.threshold,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def setup_rock_and_till(
        self, file_name, rock_erody, till_erody, rock_thresh, till_thresh, contact_width
    ):
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
        read_esri_ascii(
            file_name, grid=self.grid, name="rock_till_contact__elevation", halo=1
        )

        # Get a reference to the rock-till field
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

        # Create field for erodibility
        if "substrate__erodibility" in self.grid.at_node:
            self.erody = self.grid.at_node["substrate__erodibility"]
        else:
            self.erody = self.grid.add_zeros("node", "substrate__erodibility")

        # Create field for threshold values
        if "erosion__threshold" in self.grid.at_node:
            self.threshold = self.grid.at_node["erosion__threshold"]
        else:
            self.threshold = self.grid.add_zeros("node", "erosion__threshold")

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
        self.erody_wt[self.data_nodes] = 1.0 / (
            1.0
            + np.exp(
                -(self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                / self.contact_width
            )
        )

        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            erode_factor = self.boundary_handler[
                "PrecipChanger"
            ].get_erodibility_adjustment_factor()
            self.till_erody = self.K_till_sp * erode_factor
            self.rock_erody = self.K_rock_sp * erode_factor

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody + (1.0 - self.erody_wt) * self.rock_erody
        )

        # Calculate the effective thresholds using weighted averaging
        self.threshold[:] = (
            self.erody_wt * self.till_thresh + (1.0 - self.erody_wt) * self.rock_thresh
        )

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
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Update the erodibility and threshold field
        self.update_erodibility_and_threshold_fields()

        # Do some erosion (but not on the flooded nodes)
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
        print("Must include input file name on command line")
        sys.exit(1)

    thrt = BasicThRt(input_file=infile)
    thrt.run()


if __name__ == "__main__":
    main()
