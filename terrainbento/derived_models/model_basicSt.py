# coding: utf8
#! /usr/env/python
"""
terrainbento Model **BasicSt** program.

Erosion model program using linear diffusion and stream power. Discharge is
calculated from drainage area, infiltration capacity (a parameter), and
precipitation rate, which is a stochastic variable.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, FastscapeEroder
from terrainbento.base_class import StochasticErosionModel


class BasicSt(StochasticErosionModel):
    """
    **BasicSt** model program.

    **BasicSt** is a model program that evolves a topographic surface
    described by :math:`\eta (x,y,t)` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{q}\hat{Q}^{m}S^{n} + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable), :math:`S` is the local
    slope gradient, :math:`m` and :math:`n` are the discharge and slope
    exponents, respectively, and :math:`D` is the regolith transport parameter.

    **BasicSt** inherits from the terrainbento **StochasticErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K_q`       | ``water_erodability~stochastic`` |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`I_m`       | ``infiltration_capacity``        |
    +------------------+----------------------------------+

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.

    Model **BasicSt** models discharge and erosion across a topographic
    surface assuming (1) stochastic Poisson storm arrivals, (2) single-direction
    flow routing, and (3) Hortonian infiltration model. Includes stream-power
    erosion plus linear diffusion.

    The hydrology uses calculation of drainage area using the user-specified
    routing method. It then performs one of two options, depending on the
    user"s choice of ``opt_stochastic_duration`` (True or False).

    If the user requests stochastic duration, the model iterates through a sequence
    of storm and interstorm periods. Storm depth is drawn at random from a gamma
    distribution, and storm duration from an exponential distribution; storm
    intensity is then depth divided by duration. This sequencing is implemented by
    overriding the run_for method.

    If the user does not request stochastic duration (indicated by setting
    ``opt_stochastic_duration`` to ``False``), then the default
    (**erosion_model** base class) **run_for** method is used. Whenever
    **run_one_step** is called, storm intensity is generated at random from an
    exponential distribution with mean given by the parameter
    ``rainfall__mean_rate``. The stream power component is run for only a
    fraction of the time step duration dt, as specified by the parameter
    ``rainfall_intermittency_factor``. For example, if ``dt`` is 10 years and
    the intermittency factor is 0.25, then the stream power component is run
    for only 2.5 years.

    In either case, given a storm precipitation intensity :math:`P`, the runoff
    production rate :math:`R` [L/T] is calculated using:

    .. math::
        R = P - I (1 - \exp ( -P / I ))

    where :math:`I` is the soil infiltration capacity. At the sub-grid scale, soil
    infiltration capacity is assumed to have an exponential distribution of which
    $I$ is the mean. Hence, there are always some spots within any given grid cell
    that will generate runoff. This approach yields a smooth transition from
    near-zero runoff (when :math:`I>>P`) to :math:`R \\approx P`
    (when :math`P>>I`), without a "hard threshold."

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
        BasicSt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSt**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicSt

        Set up a parameters variable.

        >>> params = {"model_grid": "RasterModelGrid",
        ...           "dt": 1,
        ...           "output_interval": 2.,
        ...           "run_duration": 200.,
        ...           "number_of_node_rows" : 6,
        ...           "number_of_node_columns" : 9,
        ...           "node_spacing" : 10.0,
        ...           "regolith_transport_parameter": 0.001,
        ...           "water_erodability~stochastic": 0.001,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0,
        ...           "opt_stochastic_duration": False,
        ...           "number_of_sub_time_steps": 1,
        ...           "rainfall_intermittency_factor": 0.5,
        ...           "rainfall__mean_rate": 1.0,
        ...           "rainfall__shape_factor": 1.0,
        ...           "infiltration_capacity": 1.0,
        ...           "random_seed": 0}

        Construct the model.

        >>> model = BasicSt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicSt, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )
        # Get Parameters:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self._get_parameter_from_exponent("water_erodability~stochastic") * (
            self._length_factor ** ((3. * self.m) - 1)
        )  # K stochastic has units of [=] T^{m-1}/L^{3m-1}

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self._get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        # Get the infiltration-capacity parameter
        # has units length per time
        self.infilt = (self._length_factor) * self.params["infiltration_capacity"]

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Keep a reference to drainage area
        self.area = self.grid.at_node["drainage_area"]

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(self.grid, K_sp=self.K, m_sp=self.m, n_sp=self.n, discharge_name='surface_water__discharge')

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, dt):
        """Advance model ``Basic`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
        not occur.

        3. Calculates precipitation, runoff, discharge, and detachment-limited
        erosion by water.

        4. Calculates topographic change by linear diffusion.

        5. Finalizes the step using the ``ErosionModel`` base class function
        **finalize__run_one_step**. This function updates all BoundaryHandlers
        by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """
        # create and move water
        self.create_and_move_water(dt)

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
    """Execute model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print(
            (
                "To run a terrainbento model from the command line you must "
                "include input file name on command line"
            )
        )
        sys.exit(1)

    em = BasicSt(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
