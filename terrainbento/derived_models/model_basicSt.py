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


class BasicSt(StochasticErosionModel):
    r"""**BasicSt** model program.

    This model program that evolves a topographic surface,
    :math:`\eta (x,y,t)`, with the following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -K_{q}\hat{Q}^{m}S^{n}
                                           + D\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable), :math:`S` is the local
    slope gradient, :math:`m` and :math:`n` are the discharge and slope
    exponents, respectively, :math:`K` is the erodibility by water, and
    :math:`D` is the regolith transport parameter.

    Refer to
    `Barnhart et al. (2019) <https://doi.org/10.5194/gmd-12-1267-2019>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodibility=0.0001,
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
        m_sp : float, optional
            Drainage area exponent (:math:`m`). Default is 0.5.
        n_sp : float, optional
            Slope exponent (:math:`n`). Default is 1.0.
        water_erodibility : float, optional
            Water erodibility (:math:`K`). Default is 0.0001.
        regolith_transport_parameter : float, optional
            Regolith transport efficiency (:math:`D`). Default is 0.1.
        infiltration_capacity: float, optional
            Infiltration capacity (:math:`I_m`). Default is 1.0.
        **kwargs :
            Keyword arguments to pass to :py:class:`StochasticErosionModel`.
            These arguments control the discharge :math:`\hat{Q}`.

        Returns
        -------
        BasicSt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSt**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

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
        self._verify_fields(self._required_fields)

        # Get Parameters:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility
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

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Calculates precipitation, runoff, discharge, and detachment-limited
           erosion by water.

        4. Calculates topographic change by linear diffusion.

        5. Finalizes the step using the :py:mod:`ErosionModel` base class
           function **finalize__run_one_step**. This function updates all
           boundary handlers handlers by ``step`` and increments model time by
           ``step``.

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

    em = BasicSt.from_file(infile)
    em.run()


if __name__ == "__main__":
    main()
