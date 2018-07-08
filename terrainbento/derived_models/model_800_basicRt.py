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
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel

import os

class BasicRt(ErosionModel):
    """Model ``BasicRt`` program.

    Model ``BasicRt`` improves upon the ``Basic`` model by allowing for two
    lithologies. Given a spatially varying contact zone elevation, :math:`\eta_C(x,y)$)`,
    model ``BasicRt`` evolves a topographic surface described by :math:`\eta` with
    the following governing equation:

    .. math::
        :nowrap:

         \\begin{eqnarray}
             \\frac{\partial \eta}{\partial t} = - K(\eta,\eta_C) A^{1/2}S + D\nabla^2 \eta,\\
             K(\eta, \eta_C ) = w K_1 + (1 - w) K_2,\\
             w = \frac{1}{1+\exp \left( -\frac{(\eta -\eta_C )}{W_c}\right)}
         \end{eqnarray}

    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`W_c` is the contact-zone width, :math:`K_1` and :math:`K_2` are the
    erodabilities of the upper and lower lithologies, and :math:`D` is the
    regolith transport parameter. :math:`w` is a weight used to calculate the
    erodability based on the depth to the contact zone and the width of the
    contact zone.

    Refer to the ``terrainbento`` manuscript Table XX (URL here) for
    parameter symbols, names, and dimensions.where

    Model ``BasicRt`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class. Depending on the paramters provided, this model program can be used
    to run the following two ``terrainbento`` numerical models:

    1) Model ``BasicRt``: Here :math:`m` has a value of 0.5 and
    :math:`n` has a value of 1. :math:`K_{1}` is given by the parameter
    ``water_erodability~till``, :math:`K_{2}` is given by the parameter
    ``water_erodability~rock`` and :math:`D` is given by the parameter
    ``regolith_transport_parameter``.

    2) Model ``BasicRtSs``: In this model :math:`m` has a value of 1/3 and
    :math:`n` has a value of 2/3. :math:`K_{1}` is given by the parameter
    ``water_erodability~till~shear_stress``, :math:`K_{2}` is given by the
    parameter ``water_erodability~rock~shear_stress
    `` and :math:`D` is given by the parameter ``regolith_transport_parameter``.

    In both models, a value for :math:`Wc` must be given by the parameter name
    ``contact_zone__width`` and the spatially variable elevation of the contact
    elevation must be given as the file path to an ESRII ASCII format file using
    the parameter ``lithology_contact_elevation__file_name``. If topography was created with an
    input DEM, then the shape of the contact
    fill must be the same as  of
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        BasicRt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``BasicRt``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicRt

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability~rock': 0.001,
        ...           'water_erodability~till': 0.01,
        ...           'contact_zone__width': 1.0,
        ...           'lithology_contact_elevation__file_name': 'tests/data/example_contact_elevation.txt',
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        Construct the model.

        >>> model = BasicRt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
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
        K_rock_sp = self.get_parameter_from_exponent(
            "water_erodability~rock", raise_error=False
        )
        K_rock_ss = self.get_parameter_from_exponent(
            "water_erodability~rock~shear_stress", raise_error=False
        )
        K_till_sp = self.get_parameter_from_exponent(
            "water_erodability~till", raise_error=False
        )
        K_till_ss = self.get_parameter_from_exponent(
            "water_erodability~till~shear_stress", raise_error=False
        )
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        # check that a stream power and a shear stress parameter have not both been given
        # first for rock Ks
        if K_rock_sp != None and K_rock_ss != None:
            raise ValueError(
                "A parameter for both  water_erodability~rock and water_erodability~rock~shear_stress has been"
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
            raise ValueError(
                "A value for water_erodability~rock or water_erodability~rock~shear_stress  must be provided."
            )

        # Then for Till Ks
        if K_till_sp != None and K_till_ss != None:
            raise ValueError(
                "A parameter for both water_erodability~till and water_erodability~till~shear_stress has been"
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
            raise ValueError(
                "A value for water_erodability~till or water_erodability~till~shear_stress  must be provided."
            )

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till(
            self.params["lithology_contact_elevation__file_name"],
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

    def _setup_rock_and_till(self, file_name, rock_erody, till_erody, contact_width):
        """Set up lithology handling for two layers with different erodability.

        Parameters
        ----------
        file_name : string
            Name of arc-ascii format file containing elevation of contact
            position at each grid node (or NODATA)

        Read elevation of rock-till contact from an esri-ascii format file
        containing the basal elevation value at each node, create a field for
        erodability.

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
        # Read input data on rock-till contact elevation
        read_esri_ascii(
            file_name, grid=self.grid, halo=1, name="rock_till_contact__elevation"
        )

        # Get a reference to the rock-till field
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]

        # Create field for erodability
        if "substrate__erodability" in self.grid.at_node:
            self.erody = self.grid.at_node["substrate__erodability"]
        else:
            self.erody = self.grid.add_zeros("node", "substrate__erodability")

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

        # Read the erodability value of rock and till
        self.rock_erody = rock_erody
        self.till_erody = till_erody

        # Read and remember the contact zone characteristic width
        self.contact_width = contact_width

    def _update_erodability_field(self):
        """Update erodability at each node based on elevation relative to
        contact elevation.

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

        Translating these symbols into variable names:

            z = self.elev
            b = self.rock_till_contact
            D* = self.contact_width
            F = self.erody_wt
            K_till = self.till_erody
            K_rock = self.rock_erody
        """

        # Update the erodability weighting function (this is "F")
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
            ].get_erodability_adjustment_factor()
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

        # Update the erodability field
        self._update_erodability_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded, K_if_used=self.erody)

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

    brt = BasicRt(input_file=infile)
    brt.run()


if __name__ == "__main__":
    main()
