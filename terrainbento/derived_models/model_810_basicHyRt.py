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
    BasicHyRt : model object

    Examples
    --------
    This is a minimal example to demonstrate how to construct an instance
    of model **BasicHyRt**. Note that a YAML input file can be used instead of
    a parameter dictionary. For more detailed examples, including steady-
    state test examples, see the terrainbento tutorials.

    To begin, import the model class.

    >>> from terrainbento import BasicHyRt

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
    ...           'n_sp': 1.0}

    Construct the model.

    >>> model = BasicHyRt(params=params)

    Running the model with ``model.run()`` would create output, so here we
    will just run it one step.

    >>> model.run_one_step(1.)
    >>> model.model_time
    1.0

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
        self.K_rock_sp = self.get_parameter_from_exponent("water_erodability~lower")
        self.K_till_sp = self.get_parameter_from_exponent("water_erodability~upper")

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
        self.erody_br = self.grid.add_ones("node", "K_br")

        # field for rock threshold values
        self.threshold_br = self.grid.add_ones("node", "sp_crit_br")

        # Create array for erodability weighting function for BEDROCK
        self.erody_wt_br = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_and_threshold_fields(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
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
