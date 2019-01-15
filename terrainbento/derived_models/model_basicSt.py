# coding: utf8
# !/usr/env/python
"""terrainbento Model **BasicSt** program.

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

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicSt(StochasticErosionModel):
    r"""**BasicSt** model program.

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
    |:math:`K_q`       | ``water_erodability_stochastic`` |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`I_m`       | ``infiltration_capacity``        |
    +------------------+----------------------------------+

    Refer to
    `Barnhart et al. (2019) <https://www.geosci-model-dev-discuss.net/gmd-2018-204/>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

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
    fraction of the time step duration step, as specified by the parameter
    ``rainfall_intermittency_factor``. For example, if ``step`` is 10 years and
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

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability_stochastic=0.0001,
        regolith_transport_parameter=0.1,
        infiltration_capacity=1.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        clock : terrainbento Clock instance
        grid : landlab model grid instance
            The grid must have all required fields.

        **kwargs :
            Keyword arguments to pass to
            :py:class:`~terrainbento.base_class.stochastic_erosion_model.StochasticErosionModel`.

        Returns
        -------
        BasicSt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSt**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicSt
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicSt(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicSt, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(_REQUIRED_FIELDS)

        # Get Parameters:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability_stochastic
        self.infilt = infiltration_capacity

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, step):
        """Advance model ``Basic`` for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
        not occur.

        3. Calculates precipitation, runoff, discharge, and detachment-limited
        erosion by water.

        4. Calculates topographic change by linear diffusion.

        5. Finalizes the step using the ``ErosionModel`` base class function
        **finalize__run_one_step**. This function updates all BoundaryHandlers
        by ``step`` and increments model time by ``step``.

        Parameters
        ----------
        step : float
            Increment of time for which the model is run.
        """
        # create and move water
        self.create_and_move_water(step)

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Handle water erosion
        self.handle_water_erosion(step, flooded)

        # Do some soil creep
        self.diffuser.run_one_step(step)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(step)


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
