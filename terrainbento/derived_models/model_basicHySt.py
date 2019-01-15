# coding: utf8
# !/usr/env/python
"""terrainbento Model **BasicHySt** program.

Erosion model program using linear diffusion for gravitational mass transport,
and an entrainment-deposition law for water erosion and deposition. Discharge
is calculated from drainage area, infiltration capacity (a parameter), and
precipitation rate, which is a stochastic variable.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicHySt(StochasticErosionModel):
    r"""**BasicHySt** model program.

    **BasicHySt** is a model program that uses a stochastic treatment of runoff
    and discharge, and includes an erosion threshold in the water erosion law.
    THe model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -E(\hat{Q}) + D_s(\hat{Q}) + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable), :math:`E` is the bed erosion
    (entrainment) rate due to fluid entrainment, :math:`D_s` is the deposition
    rate of sediment settling out of active transport, and :math:`D` is the
    regolith transport parameter.

    **BasicHySt** inherits from the terrainbento **StochasticErosionModel**
    base class. In addition to the parameters required by the base class, models
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
    |:math:`V_s`       | ``v_s``                          |
    +------------------+----------------------------------+
    |:math:`F_f`       | ``fraction_fines``               |
    +------------------+----------------------------------+
    |:math:`\phi`      | ``sediment_porosity``            |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`I_m`       | ``infiltration_capacity``        |
    +------------------+----------------------------------+

    Refer to
    `Barnhart et al. (2019) <https://www.geosci-model-dev-discuss.net/gmd-2018-204/>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    """

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability_stochastic=0.0001,
        regolith_transport_parameter=0.1,
        settling_velocity=0.001,
        infiltration_capacity=1.0,
        sediment_porosity=0.3,
        fraction_fines=0.5,
        solver="basic",
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
        BasicHySt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHySt**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicHySt
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicHySt(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicHySt, self).__init__(clock, grid, **kwargs)

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

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            F_f=fraction_fines,
            phi=sediment_porosity,
            v_s=settling_velocity,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, step):
        """Advance model for one time-step of duration step."""
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
