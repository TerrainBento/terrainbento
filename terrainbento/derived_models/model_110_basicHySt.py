# coding: utf8
#! /usr/env/python
"""
model_110_basicHySt.py: erosion model with stochastic
rainfall and hybrid alluvial incision.

Model 110 BasicHySt

The hydrology aspect models discharge and erosion across a topographic
surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
flow routing, and (3) Hortonian infiltration model. Includes stream-power
erosion plus linear diffusion.

The hydrology uses calculation of drainage area using the standard "D8"
approach (assuming the input grid is a raster; "DN" if not), then modifies it
by running a lake-filling component. It then iterates through a sequence of
storm and interstorm periods. Storm depth is drawn at random from a gamma
distribution, and storm duration from an exponential distribution; storm
intensity is then depth divided by duration. Given a storm precipitation
intensity $P$, the runoff production rate $R$ [L/T] is calculated using:

$R = P - I (1 - \exp ( -P / I ))$

where $I$ is the soil infiltration capacity. At the sub-grid scale, soil
infiltration capacity is assumed to have an exponential distribution of which
$I$ is the mean. Hence, there are always some spots within any given grid cell
that will generate runoff. This approach yields a smooth transition from
near-zero runoff (when $I>>P$) to $R \approx P$ (when $P>>I$), without a
"hard threshold."

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution, LinearDiffuser, HybridAlluvium

"""

import numpy as np

from landlab.components import LinearDiffuser, ErosionDeposition
from terrainbento.base_class import StochasticErosionModel


class BasicHySt(StochasticErosionModel):
    """
    A BasicHySt computes erosion using (1) hybrid alluvium river erosion,
    (2) linear nhillslope diffusion, and
    (3) generation of a random sequence of runoff events across a topographic
    surface.
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
        BasicHySt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHySt**. Note that a YAML input file can be used instead
        of a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicHySt

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
        ...           'rainfall_intermittency_factor': 0.5,
        ...           'rainfall__mean_rate': 1.0,
        ...           'rainfall__shape_factor': 1.0,
        ...           'infiltration_capacity': 1.0,
        ...           'random_seed': 0,
        ...           'v_s': 0.01,
        ...           'fraction_fines': 0.1,
        ...           'sediment_porosity': 0.3,
        ...           'solver': 'adaptive'}

        Construct the model.

        >>> model = BasicHySt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicHySt, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )

        # Get Parameters:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self.get_parameter_from_exponent("water_erodability~stochastic") * (
            self._length_factor ** ((3. * self.m) - 1)
        )  # K stochastic has units of [=] T^{m-1}/L^{3m-1}

        regolith_transport_parameter = (
            self._length_factor ** 2
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # L^2/T

        v_s = (self._length_factor) * self.get_parameter_from_exponent(
            "v_s"
        )  # has units length per time

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Get the infiltration-capacity parameter
        # has units length per time
        self.infilt = (self._length_factor) * self.params["infiltration_capacity"]

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Keep a reference to drainage area
        self.area = self.grid.at_node["drainage_area"]

        # Handle solver option
        solver = self.params.get("solver", "basic")

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            F_f=self.params["fraction_fines"],
            phi=self.params["sediment_porosity"],
            v_s=v_s,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
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

    em = BasicHySt(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
