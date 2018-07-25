# coding: utf8
#! /usr/env/python
"""
model_300_basicStVs.py: models discharge and erosion across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) a variable-source-area (VSA) runoff-generation model.

Model 300 BasicStVs

This model combines linear diffusion and basic stream power with stochastic
variable source area (VSA) hydrology. It inherits from the ErosionModel
class. It calculates drainage area using the standard "D8" approach (assuming
the input grid is a raster; "DN" if not), then modifies it by running a
lake-filling component. It then iterates through a sequence of storm and
interstorm periods. Storm depth is drawn at random from a gamma distribution,
and storm duration from an exponential distribution; storm intensity is then
depth divided by duration. Given a storm precipitation intensity $P$, the
discharge $Q$ [L$^3$/T] is calculated using:

$Q = PA - T\lambda S [1 - \exp (-PA/T\lambda S) ]$

where $T$ is the soil transmissivity and $\lambda$ is cell width.

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution, StreamPowerEroder, LinearDiffuser

"""

import numpy as np

from landlab.components import StreamPowerEroder, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel


class BasicStVs(StochasticErosionModel):
    """
    A BasicStVs generates a random sequency of
    runoff events across a topographic surface, calculating the resulting
    water discharge at each node.
    """

    def __init__(self, input_file=None, params=None, OutputWriters=None):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        BasicStVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicStVs**. Note that a YAML input file can be used instead
        of a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicStVs

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability~stochastic': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'opt_stochastic_duration': False,
        ...           'number_of_sub_time_steps': 1,
        ...           'daily_rainfall_intermittency_factor': 0.5,
        ...           'daily_rainfall__mean_intensity': 1.0,
        ...           'daily_rainfall__precipitation_shape_factor': 1.0,
        ...           'infiltration_capacity': 1.0,
        ...           'random_seed': 0,
        ...           'soil__initial_thickness': 2.0,
        ...           'hydraulic_conductivity': 0.1}

        Construct the model.

        >>> model = BasicStVs(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicStVs, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )
        # Get Parameters:
        K_sp = self.get_parameter_from_exponent("water_erodability~stochastic")

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        soil_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length
        K_hydraulic_conductivity = (self._length_factor) * self.params[
            "hydraulic_conductivity"
        ]  # has units length per time

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Add a field for subsurface discharge
        self.qss = self.grid.add_zeros("node", "subsurface_water__discharge")

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.trans = K_hydraulic_conductivity * soil_thickness

        if self.trans <= 0.0:
            raise ValueError("BasicStVs: Transmissivity must be > 0")

        self.tlam = self.trans * self.grid._dx  # assumes raster

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Keep a reference to drainage area and steepest-descent slope
        self.area = self.grid.at_node["drainage_area"]
        self.slope = self.grid.at_node["topographic__steepest_slope"]

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
            self.grid,
            use_Q=self.discharge,
            K_sp=K_sp,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""

        # Here's the total (surface + subsurface) discharge
        pa = self.rain_rate * self.area

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = self.tlam * self.slope[np.where(self.slope > 0.0)[0]]

        # Subsurface discharge: zero where slope is flat
        self.qss[np.where(self.slope <= 0.0)[0]] = 0.0
        self.qss[np.where(self.slope > 0.0)[0]] = tls * (
            1.0 - np.exp(-pa[np.where(self.slope > 0.0)[0]] / tls)
        )

        # Surface discharge = total minus subsurface
        #
        # Note that roundoff errors can sometimes produce a tiny negative
        # value when qss and pa are close; make sure these are set to 0
        self.discharge[:] = pa - self.qss
        self.discharge[self.discharge < 0.0] = 0.0

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

        # Handle water erosion
        self.handle_water_erosion(dt, flooded)

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

    dm = BasicStVs(input_file=infile)
    dm.run()


if __name__ == "__main__":
    main()
