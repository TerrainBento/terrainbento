#! /usr/env/python
"""
model_808_basicDdRt.py: erosion model using linear diffusion, stream
power with a smoothed threshold that increases with incision depth, discharge
proportional to drainage area, and two lithologies: rock and till.

Model 808 BasicDdRt

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerSmoothThresholdEroder, LinearDiffuser

"""

import sys
import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, LinearDiffuser
from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class BasicDdRt(ErosionModel):
    """
    A BasicDdRt computes erosion using linear diffusion, stream
    power with a smoothed threshold, Q~A, and two lithologies: rock and till.
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
        BasicDdRt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDdRt**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicDdRt

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
        ...           'water_erosion_rule~substrate~threshold___parameter': 0.2,
        ...           'water_erosion_rule~substrate~threshold__depth_derivative_of_parameter': 0.001,
        ...           'contact_zone__width': 1.0,
        ...           'lithology_contact_elevation__file_name': 'tests/data/example_contact_elevation.txt',
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        Construct the model.

        >>> model = BasicDdRt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicDdRt, self).__init__(
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
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")
        self.threshold_value = self._length_factor * self.get_parameter_from_exponent(
            "water_erosion_rule~substrate~threshold___parameter"
        )  # has units length/time

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock_sp
        self.till_erody = self.K_till_sp

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros("node", "erosion__threshold")
        self.threshold[:] = self.threshold_value

        # Instantiate a StreamPowerSmoothThresholdEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.erody,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            threshold_sp=self.threshold,
        )

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params["water_erosion_rule~substrate~threshold__depth_derivative_of_parameter"]

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

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_field(self):
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

    def _update_erosion_threshold_values(self):
        """Updates the erosion threshold at each node based on cumulative
        erosion so far."""

        # Set the erosion threshold.
        #
        # Note that a minus sign is used because cum ero depth is negative for
        # erosion, positive for deposition.
        # The second line handles the case where there is growth, in which case
        # we want the threshold to stay at its initial value rather than
        # getting smaller.
        cum_ero = self.grid.at_node["cumulative_erosion__depth"]
        cum_ero[:] = self.z - self.grid.at_node["initial_topographic__elevation"]
        self.threshold[:] = self.threshold_value - (
            self.thresh_change_per_depth * cum_ero
        )
        self.threshold[self.threshold < self.threshold_value] = self.threshold_value

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
        self._update_erodability_field()

        # Calculate the new threshold values given cumulative erosion
        self._update_erosion_threshold_values()

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

    thrt = BasicDdRt(input_file=infile)
    thrt.run()


if __name__ == "__main__":
    main()
