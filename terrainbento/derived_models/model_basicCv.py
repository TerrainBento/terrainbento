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


class BasicCv(ErosionModel):
    r"""**BasicCv** model program.

    This model program evolves a topographic surface, :math:`\eta`, with the
    following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -KQ^{m}S^{n} + D\nabla^2 \eta

    where :math:`K` is the fluviel erodibility coefficient, :math:`Q` is the
    local stream discharge, :math:`S` is the local slope, :math:`m` and
    :math:`n` are the discharge and slope exponent parameters, and :math:`D` is
    the regolith transport parameter.

    This model also has a basic parameterization of climate change such that
    :math:`K` varies through time. Between model run onset and a time at
    which the climate becomes constant, the value of :math:`K` linearly
    changes from :math:`fK` to :math:`K`, at which point it remains at
    :math:`K` for the remainder of the modeling time period.

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
        climate_factor=0.5,
        climate_constant_date=0.0,
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
        climate_factor : float, optional.
             Default is 0.5.(:math:`f` )
        climate_constant_date : float, optional.
            Model time at which climate becomes constant (:math:`T_s`) and
            water erodibility stabilizes at a  value of :math:`K`. Default
            is 0.0.
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        Basic : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``Basic``. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

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
        self._verify_fields(self._required_fields)

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
            water_erodibility * self.climate_factor,
            water_erodibility,
            water_erodibility,
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

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Updates detachment-limited erosion based on climate.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the :py:mod:`ErosionModel` base class
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

    ldsp = BasicCv.from_file(infile)
    ldsp.run()


if __name__ == "__main__":
    main()
