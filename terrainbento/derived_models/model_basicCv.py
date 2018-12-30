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
    """**BasicCv** model program.

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

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.

    """

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------


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
        >>> from terrainbento import Clock, Basic
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = Basic(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        >>> params = {"model_grid": "RasterModelGrid",
        ...           "clock": {"step": 1,
        ...                     "output_interval": 2.,
        ...                     "stop": 200.},
        ...           "number_of_node_rows" : 6,
        ...           "number_of_node_columns" : 9,
        ...           "node_spacing" : 10.0,
        ...           "regolith_transport_parameter": 0.001,
        ...           "water_erodability": 0.001,
        ...           "climate_factor": 1.0,
        ...           "climate_constant_date": 1.0,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0}

        Construct the model.

        >>> model = BasicCv(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicCv, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        K_sp = self._get_parameter_from_exponent("water_erodability") * (
            self._length_factor ** (1. - (2. * self.m))
        )
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self._get_parameter_from_exponent("regolith_transport_parameter")

        self.climate_factor = self.params["climate_factor"]
        self.climate_constant_date = self.params["climate_constant_date"]

        time = [0, self.climate_constant_date, self.params["clock"]["stop"]]
        K = [K_sp * self.climate_factor, K_sp, K_sp]
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
