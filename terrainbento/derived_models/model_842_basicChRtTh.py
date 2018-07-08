#! /usr/env/python
"""
model_840_basicChRtTh.py: erosion model using cubic diffusion, basic stream
power with a threshold and spatially varying K and two bedrock units, and discharge
proportional to drainage area.

Model 842 BasicChRtTh

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, CubicNonLinearDiffuser
"""

import sys
import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class BasicChRtTh(ErosionModel):
    """
    A BasicChRt model computes erosion using cubic diffusion, basic stream
    power with two rock units, and Q~A.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicChRt model."""

        # Call ErosionModel's init
        super(BasicChRtTh, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        self.contact_width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length
        self.K_rock_sp = self.get_parameter_from_exponent("water_erodability~rock")
        self.K_till_sp = self.get_parameter_from_exponent("water_erodability~till")
        rock_erosion__threshold = self.get_parameter_from_exponent(
            "rock_erosion__threshold"
        )
        till_erosion__threshold = self.get_parameter_from_exponent(
            "till_erosion__threshold"
        )
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock_sp
        self.till_erody = self.K_till_sp

        # Save the threshold values for rock and till
        self.rock_thresh = rock_erosion__threshold
        self.till_thresh = till_erosion__threshold

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Instantiate a StreamPowerSmoothThresholdEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.erody,
            threshold_sp=self.threshold,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=self.params["slope_crit"],
            nterms=7,
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

        # Create field for erodability
        self.erody = self.grid.add_zeros("node", "substrate__erodability")

        # Create field for threshold values
        self.threshold = self.grid.add_zeros("node", "erosion__threshold")

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_and_threshold_fields(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
        """
        # Update the erodability weighting function (this is "F")
        D_over_D_star = (
            self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes]
        ) / self.contact_width

        # truncate D_over_D star to remove potential for overflow in exponent
        D_over_D_star[D_over_D_star < -100.0] = -100.0
        D_over_D_star[D_over_D_star > 100.0] = 100.0

        self.erody_wt[self.data_nodes] = 1.0 / (1.0 + np.exp(-D_over_D_star))

        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            erode_factor = self.boundary_handler[
                "PrecipChanger"
            ].get_erodability_adjustment_factor()
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
        # Direct and accumulate flow
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
        self.diffuser.run_one_step(
            dt, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

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

    chrt = BasicChRtTh(input_file=infile)
    chrt.run()


if __name__ == "__main__":
    main()
