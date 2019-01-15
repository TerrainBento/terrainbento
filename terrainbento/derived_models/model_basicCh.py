# coding: utf8
# !/usr/env/python
"""terrainbento **BasicCh** model program.

Erosion model program using cubic diffusion, basic stream power, and discharge
proportional to drainage area.


Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `TaylorNonLinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.taylor_nonlinear_hillslope_flux.html>`_
"""

import numpy as np

from landlab.components import FastscapeEroder, TaylorNonLinearDiffuser
from terrainbento.base_class import ErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicCh(ErosionModel):
    r"""**BasicCh** model program.

    **BasicCh** is a model program that evolves a topographic surface described
    by :math:`\eta` with the following governing equations:


    .. math::

        \\frac{\partial \eta}{\partial t} = -KA^{m}S^{n} + \\nabla^2 q_h

        q_h = -DS \left[ 1 + \left( \\frac{S}{S_c} \\right)^2 +  \left( \\frac{S}{S_c} \\right)^4 + ... \left( \\frac{S}{S_c} \\right)^{2(N-1)} \\right]


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`D` is the regolith
    transport efficiency, and :math:`S_c` is the critical slope. :math:`q_h`
    represents the hillslope sediment flux per unit width. :math:`N` is the
    number of terms in the Taylor Series expansion. :math:`N` is set at a
    default value of 11 but can be modified by a user.

    The **BasicCh** program inherits from the terrainbento **ErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K`         | ``water_erodability``            |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`S_c`       | ``critical_slope``               |
    +------------------+----------------------------------+
    |:math:`N`         | ``number_of_taylor_terms``       |
    +------------------+----------------------------------+

    refer to
    `Barnhart et al. (2019) <https://www.geosci-model-dev-discuss.net/gmd-2018-204/>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    """

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        critical_slope=0.3,
        number_of_taylor_terms=11,
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
            :py:class:`~terrainbento.base_class.erosion_model.ErosionModel`.

        Returns
        -------
        BasicCh : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicCh**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicCh
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicCh(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel"s init
        super(BasicCh, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(_REQUIRED_FIELDS)

        # Get Parameters and convert units if necessary:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability

        regolith_transport_parameter = regolith_transport_parameter

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a NonLinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=critical_slope,
            nterms=number_of_taylor_terms,
        )

    def run_one_step(self, step):
        """Advance model **BasicCh** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by nonlinear diffusion.

        6. Finalizes the step using the **ErosionModel** base class function
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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            self.eroder.K = (
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(step, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(
            step, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

        # Finalize the run_one_step_method
        self.finalize__run_one_step(step)


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    cdsp = BasicCh(input_file=infile)
    cdsp.run()


if __name__ == "__main__":
    main()
