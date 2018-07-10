#! /usr/env/python
"""
model_802_basicThRt.py: erosion model using linear diffusion, stream
power with a smoothed threshold, discharge proportional to drainage area, and
two lithologies: rock and till.

Model 802 BasicRtTh

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, LinearDiffuser

"""

import sys
import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, LinearDiffuser
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class BasicRtTh(ErosionModel):
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
    BasicRtTh : model object

    Examples
    --------
    This is a minimal example to demonstrate how to construct an instance
    of model **BasicRtTh**. Note that a YAML input file can be used instead of
    a parameter dictionary. For more detailed examples, including steady-
    state test examples, see the terrainbento tutorials.

    To begin, import the model class.

    >>> from terrainbento import BasicRtTh

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
    ...           'water_erosion_rule~upper~threshold___parameter': 0.1,
    ...           'water_erosion_rule~lower~threshold___parameter': 0.2,
    ...           'contact_zone__width': 1.0,
    ...           'lithology_contact_elevation__file_name': 'tests/data/example_contact_elevation.txt',
    ...           'm_sp': 0.5,
    ...           'n_sp': 1.0}

    Construct the model.

    >>> model = BasicRtTh(params=params)

    Running the model with ``model.run()`` would create output, so here we
    will just run it one step.

    >>> model.run_one_step(1.)
    >>> model.model_time
    1.0

    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicRtTh."""
        # Call ErosionModel's init
        super(BasicRtTh, self).__init__(
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
        rock_erosion__threshold = self.get_parameter_from_exponent(
            "water_erosion_rule~lower~threshold___parameter"
        )
        till_erosion__threshold = self.get_parameter_from_exponent(
            "water_erosion_rule~upper~threshold___parameter"
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

    thrt = BasicRtTh(input_file=infile)
    thrt.run()


if __name__ == "__main__":
    main()
