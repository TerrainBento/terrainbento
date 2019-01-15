# coding: utf8
# !/usr/env/python
"""terrainbento **BasicCv** model program.

Erosion model program using linear diffusion, stream power, and discharge
proportional to drainage area with climate change.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np
from scipy.interpolate import interp1d

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicCv(ErosionModel):
    r"""**BasicCv** model program.

    **BasicCv** is a model program that evolves a topographic surface described
    by :math:`\eta` with the following governing equation:


    .. math::

        \\frac{\partial \eta}{\partial t} = -KA^{m}S^{n} + D\\nabla^2 \eta


    where :math:`K` is the fluviel erodability coefficient, :math:`A` is the
    local drainage area, :math:`S` is the local slope, :math:`m` and :math:`n`
    are the drainage area and slope exponent parameters, and :math:`D` is the
    regolith transport parameter.

    This model also has a basic parameterization of climate change such that
    :math:`K` varies through time. Between model run onset and a time at
    which the climate becomes constant, the value of :math:`K` linearly
    changes from :math:`fK` to :math:`K`, at which point it remains at :math:`K`
    for the remainder of the modeling time period.

    The **BasicCv** program inherits from the terrainbento **ErosionModel** base
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
    |:math:`f`         | ``climate_factor``               |
    +------------------+----------------------------------+
    |:math:`T_s`       | ``climate_constant_date``        |
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
        climate_factor=0.5,
        climate_constant_date=0.,
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
        Basic : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``Basic``. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicCv
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicCv(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicCv, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(_REQUIRED_FIELDS)

        self.m = m_sp
        self.n = n_sp
        self.climate_factor = climate_factor
        self.climate_constant_date = climate_constant_date

        time = [
            0,
            self.climate_constant_date,
            self.clock.stop + self.clock.step,
        ]
        K = [
            water_erodability * self.climate_factor,
            water_erodability,
            water_erodability,
        ]
        self.K_through_time = interp1d(time, K)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=K[0],
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

        3. Updates detachment-limited erosion based on climate.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by linear diffusion.

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

        # Update erosion based on climate
        self.eroder.K = float(self.K_through_time(self.model_time))

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(step, flooded_nodes=flooded)

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

    ldsp = BasicCv(input_file=infile)
    ldsp.run()


if __name__ == "__main__":
    main()
