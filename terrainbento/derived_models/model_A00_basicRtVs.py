#! /usr/env/python
"""
model_A00_basicVsRt.py: erosion model using linear diffusion, basic stream
power with 2 lithologies (rock and till), and discharge proportional to
effective drainage area.

Model A00 BasicRtVs

Landlab components used: FlowRouter, StreamPowerEroder, LinearDiffuser
"""

import sys
import numpy as np

from landlab.components import StreamPowerEroder, LinearDiffuser
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class BasicRtVs(ErosionModel):
    """
    A BasicRtVs computes erosion using linear diffusion, basic stream
    power with 2 lithologies, and Q ~ A exp( -b S / A).
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
        BasicRtVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicRtVs**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicRtVs

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability~lower': 0.001,
        ...           'water_erodability~upper': 0.01,
        ...           'contact_zone__width': 1.0,
        ...           'lithology_contact_elevation__file_name': 'tests/data/example_contact_elevation.txt',
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'recharge_rate': 0.5,
        ...           'soil__initial_thickness': 2.0,
        ...           'hydraulic_conductivity': 0.1}

        Construct the model.

        >>> model = BasicRtVs(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicRtVs, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )
        self.contact_width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length
        self.K_rock_sp = self.get_parameter_from_exponent("water_erodability~lower")
        self.K_till_sp = self.get_parameter_from_exponent("water_erodability~upper")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        recharge_rate = (self._length_factor) * self.params[
            "recharge_rate"
        ]  # has units length per time
        soil_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length
        K_hydraulic_conductivity = (self._length_factor) * self.params[
            "hydraulic_conductivity"
        ]  # has units length per time

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock_sp
        self.till_erody = self.K_till_sp

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-area parameter
        self.sat_param = (K_hydraulic_conductivity * soil_thickness * self.grid.dx) / (
            recharge_rate
        )

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
            self.grid,
            K_sp=self.erody,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            use_Q=self.eff_area,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def _calc_effective_drainage_area(self):
        """Calculate and store effective drainage area.

        Effective drainage area is defined as:

        $A_{eff} = A \exp ( \alpha S / A) = A R_r$

        where $S$ is downslope-positive steepest gradient, $A$ is drainage
        area, $R_r$ is the runoff ratio, and $\alpha$ is the saturation
        parameter.
        """
        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        cores = self.grid.core_nodes
        self.eff_area[cores] = area[cores] * (
            np.exp(-self.sat_param * slope[cores] / area[cores])
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

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_field(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
        """
        # Update the erodability weighting function (this is "F")
        core = self.grid.core_nodes
        if self.contact_width > 0.0:
            self.erody_wt[core] = 1.0 / (
                1.0
                + np.exp(
                    -(self.z[core] - self.rock_till_contact[core]) / self.contact_width
                )
            )
        else:
            self.erody_wt[core] = 0.0
            self.erody_wt[np.where(self.z > self.rock_till_contact)[0]] = 1.0

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

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Direct and accumulate flow
        self.flow_accumulator.run_one_step()

        # Update effective runoff ratio
        self._calc_effective_drainage_area()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Zero out effective area in flooded nodes
        self.eff_area[flooded] = 0.0

        # Update the erodability field
        self._update_erodability_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt)

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

    vsrt = BasicRtVs(input_file=infile)
    vsrt.run()


if __name__ == "__main__":
    main()
