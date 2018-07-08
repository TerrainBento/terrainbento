#! /usr/env/python
"""
model_810_basicHyRt.py: erosion model using linear diffusion, the hybrid
alluvium stream erosion model, discharge proportional to drainage area, and
two lithologies: rock and till.

Model 810 BasicHyRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         Space, LinearDiffuser

IMPORTANT: This model allows changes in erodability and threshold for bedrock
abd sediment INDEPENDENTLY, meaning that weighting functions etc. exist for
both.
"""

import sys
import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class BasicHyRt(ErosionModel):
    """
    A BasicHyRt computes erosion using linear diffusion, hybrid alluvium
    stream erosion, Q~A, and two lithologies: rock and till.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicHyRt."""

        # Call ErosionModel's init
        super(BasicHyRt, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        self.contact_width = (
            self._length_factor * self.params["contact_zone__width"]
        )  # L
        self.K_rock_sp = self.get_parameter_from_exponent("water_erodability~rock")
        self.K_till_sp = self.get_parameter_from_exponent("water_erodability~till")

        regolith_transport_parameter = (
            self._length_factor ** 2
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        v_sc = self.get_parameter_from_exponent(
            "v_sc"
        )  # normalized settling velocity. Unitless.

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody_br = self.K_rock_sp
        self.till_erody_br = self.K_till_sp

        # Save the threshold values for rock and till
        self.rock_thresh_br = 0.
        self.till_thresh_br = 0.

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Handle solver option
        try:
            solver = self.params["solver"]
        except:
            solver = "original"

        # Instantiate an ErosionDeposition ("hybrid") component
        self.eroder = ErosionDeposition(
            self.grid,
            K="K_br",
            F_f=self.params["F_f"],
            phi=self.params["phi"],
            v_s=v_sc,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            method="simple_stream_power",
            discharge_method="drainage_area",
            area_field="drainage_area",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def _setup_rock_and_till(self):
        """Set up fields to handle for two layers with different erodability."""
        file_name = self.params["lithology_contact_elevation__file_name"]
        # Read input data on rock-till contact elevation
        read_esri_ascii(
            file_name, grid=self.grid, name="rock_till_contact__elevation", halo=1
        )

        # Get a reference to the rock-till field
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

        # Create field for rock erodability
        if "K_br" in self.grid.at_node:
            self.erody_br = self.grid.at_node["K_br"]
        else:
            self.erody_br = self.grid.add_ones("node", "K_br")
            self.erody_br[:] = self.rock_erody_br

        # field for rock threshold values
        if "sp_crit_br" in self.grid.at_node:
            self.threshold_br = self.grid.at_node["sp_crit_br"]
        else:
            self.threshold_br = self.grid.add_ones("node", "sp_crit_br")
            self.threshold_br[:] = self.rock_thresh_br

        # Create array for erodability weighting function for BEDROCK
        self.erody_wt_br = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_and_threshold_fields(self):
        """Update erodability and threshold at each node based on elevation
        relative to contact elevation.

        To promote smoothness in the solution, the erodability at a given point
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
               zone", where the effective erodability is a mixture of the two.
               If the surface is well above this contact zone, then F = 1. If
               it's well below the contact zone, then F = 0.
            3. Set the erodability using F:
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

        # Update the erodability weighting function (this is "F")
        self.erody_wt_br[self.data_nodes] = 1.0 / (
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
            ].get_erodability_adjustment_factor()
            self.till_erody_br = self.K_till_sp * erode_factor
            self.rock_erody_br = self.K_rock_sp * erode_factor

        # Calculate the effective BEDROCK erodibilities using weighted averaging
        self.erody_br[:] = (
            self.erody_wt_br * self.till_erody_br
            + (1.0 - self.erody_wt_br) * self.rock_erody_br
        )

        # Calculate the effective BEDROCK thresholds using weighted averaging
        self.threshold_br[:] = (
            self.erody_wt_br * self.till_thresh_br
            + (1.0 - self.erody_wt_br) * self.rock_thresh_br
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

        # Update the erodability and threshold field
        self._update_erodability_and_threshold_fields()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    thrt = BasicHyRt(input_file=infile)
    thrt.run()


if __name__ == "__main__":
    main()
