#! /usr/env/python
"""``terrainbento`` Model ``BasicRt`` program.

Erosion model program using linear diffusion, stream power with spatially
varying erodability based on two bedrock units, and discharge proportional to
drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import sys
import numpy as np

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicRt(ErosionModel):
    """Model ``BasicRt`` program.

    Model ``BasicRt`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{w}A^{m}S^{n} + D\\nabla^2 \eta

    where :math:`A` is the local drainage area and :math:`S` is the local slope.
    Refer to the ``terrainbento`` manuscript Table XX (URL here) for parameter
    symbols, names, and dimensions.

    :math:`K_{w}` is the erodability of the ground substrate by water and is
    permitted to vary spatially through the file **rock_till_file__name**.

    **contact_zone__width**

    Model ``BasicRt`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class. Depending on the values of :math:`K_{w}`, :math:`D`, :math:`m`
    and, :math:`n` this model program can be used to run the following two
    ``terrainbento`` numerical models:

    1) Model ``BasicRt``: Here :math:`m` has a value of 0.5 and
    :math:`n` has a value of 1. :math:`K_{w}` is given by the parameter
    ``water_erodibility`` and :math:`D` is given by the parameter
    ``regolith_transport_parameter``.

    2) Model ``BasicRtSs``: In this model :math:`m` has a value of 1/3,
    :math:`n` has a value of 2/3, and :math:`K_{w}` is given by the
    parameter ``water_erodibility~shear_stress``.
    """

    """
    A BasicRt model computes erosion using linear diffusion, basic stream
    power with two rock units, and Q~A.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicRt model."""

        # Call ErosionModel's init
        super(BasicRt, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        contact_zone__width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length
        K_rock_sp = self.get_parameter_from_exponent("water_erodability~rock", raise_error=False)
        K_rock_ss = self.get_parameter_from_exponent("water_erodability~rock~shear_stress", raise_error=False)
        K_till_sp = self.get_parameter_from_exponent("water_erodability~till", raise_error=False)
        K_till_ss = self.get_parameter_from_exponent("water_erodability~till~shear_stress", raise_error=False)
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        # check that a stream power and a shear stress parameter have not both been given
        # first for rock Ks
        if K_rock_sp != None and K_rock_ss != None:
            raise ValueError(
                "A parameter for both K_rock_sp and K_rock_ss has been"
                "provided. Only one of these may be provided"
            )
        elif K_rock_sp != None or K_rock_ss != None:
            if K_rock_sp != None:
                self.K_rock = K_rock_sp
            else:
                self.K_rock = (
                    self._length_factor ** (1. / 3.)
                ) * K_rock_ss  # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError("A value for K_rock_sp or K_rock_ss  must be provided.")

        # Then for Till Ks
        if K_till_sp != None and K_till_ss != None:
            raise ValueError(
                "A parameter for both K_till_sp and K_rock_ss has been"
                "provided. Only one of these may be provided"
            )
        elif K_till_sp != None or K_till_ss != None:
            if K_till_sp != None:
                self.K_till = K_till_sp
            else:
                self.K_till = (
                    self._length_factor ** (1. / 3.)
                ) * K_till_ss  # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError("A value for K_till_sp or K_till_ss  must be provided.")

        # Set up rock-till
        self.setup_rock_and_till(
            self.params["rock_till_file__name"],
            self.K_rock,
            self.K_till,
            contact_zone__width,
        )

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.erody,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def setup_rock_and_till(self, file_name, rock_erody, till_erody, contact_width):
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
        read_esri_ascii(
            file_name, grid=self.grid, halo=1, name="rock_till_contact__elevation"
        )

        # Get a reference to the rock-till field
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

        # Create field for erodibility
        if "substrate__erodibility" in self.grid.at_node:
            self.erody = self.grid.at_node["substrate__erodibility"]
        else:
            self.erody = self.grid.add_zeros("node", "substrate__erodibility")

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
            self.till_erody = self.K_till * erode_factor
            self.rock_erody = self.K_rock * erode_factor

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody + (1.0 - self.erody_wt) * self.rock_erody
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

        # Update the erodibility field
        self.update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded, K_if_used=self.erody)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main(): #pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    brt = BasicRt(input_file=infile)
    brt.run()


if __name__ == "__main__":
    main()
