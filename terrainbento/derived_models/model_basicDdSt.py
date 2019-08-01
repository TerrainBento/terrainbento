# coding: utf8
# !/usr/env/python
"""terrainbento **BasicDdSt** model program.

Erosion model program using linear diffusion, smoothly thresholded stream
power, and stochastic discharge with a smoothed infiltration capacity
threshold. The program differs from BasicStTh in that the threshold value
depends on cumulative incision depth, and so can vary in space and time.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerSmoothThresholdEroder`
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, StreamPowerSmoothThresholdEroder
from terrainbento.base_class import StochasticErosionModel


class BasicDdSt(StochasticErosionModel):
    r"""**BasicDdSt** model program.

    This model program uses a stochastic treatment of runoff and discharge, and
    includes an erosion threshold in the water erosion law. The threshold
    depends on cumulative incision depth, and therefore can vary in space and
    time. It combines models :py:class:`BasicDd` and :py:class:`BasicSt`.

    The model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -\left[K_{q}\hat{Q}^{m}S^{n}
                     - \omega_{ct} \left(1-e^{-K_{q}\hat{Q}^{m}S^{n}
                       / \omega_{ct}}\right)\right)]
                       + D \nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable) and :math:`S` is the local
    slope gradient. :math:`m` and :math:`n` are the discharge and slope
    exponent, respectively, :math:`\omega_c` is the critical stream power
    required for erosion to occur, :math:`K` is the erodibility by water, and
    :math:`D` is the regolith transport parameter.

    :math:`\omega_{ct}` may change through time as it increases with cumulative
    incision depth:

    .. math::

        \omega_{ct}\left(x,y,t\right) = \mathrm{max}\left(\omega_c
                                 + b D_I\left(x, y, t\right), \omega_c \right)

    where :math:`\omega_c` is the threshold when no incision has taken place,
    :math:`b` is the rate at which the threshold increases with incision depth,
    and :math:`D_I` is the cumulative incision depth at location
    :math:`\left(x,y\right)` and time :math:`t`.

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
        water_erosion_rule__threshold=0.01,
        water_erosion_rule__thresh_depth_derivative=0.0,
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
        water_erosion_rule__threshold : float, optional
            Erosion rule threshold when no erosion has occured
            (:math:`\omega_c`). Default is 0.01.
        water_erosion_rule__thresh_depth_derivative : float, optional
            Rate of increase of water erosion threshold as increased incision
            occurs (:math:`b`). Default is 0.0.
        infiltration_capacity: float, optional
            Infiltration capacity (:math:`I_m`). Default is 1.0.
        **kwargs :
            Keyword arguments to pass to :py:class:`StochasticErosionModel`.
            These arguments control the discharge :math:`\hat{Q}`.

        Returns
        -------
        BasicDdSt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDdSt**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicDdSt
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicDdSt(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicDdSt, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Get Parameters:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility
        self.threshold_value = water_erosion_rule__threshold
        self.thresh_change_per_depth = (
            water_erosion_rule__thresh_depth_derivative
        )
        self.infilt = infiltration_capacity

        if float(self.n) != 1.0:
            raise ValueError("Model only supports n equals 1.")

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros(
            "node", "water_erosion_rule__threshold"
        )
        self.threshold[:] = self.threshold_value

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            m_sp=self.m,
            n_sp=self.n,
            K_sp=self.K,
            use_Q="surface_water__discharge",
            threshold_sp=self.threshold,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def update_threshold_field(self):
        """Update the threshold based on cumulative erosion depth."""
        cum_ero = self.grid.at_node["cumulative_elevation_change"]
        cum_ero[:] = (
            self.z - self.grid.at_node["initial_topographic__elevation"]
        )
        self.threshold[:] = self.threshold_value - (
            self.thresh_change_per_depth * cum_ero
        )
        self.threshold[
            self.threshold < self.threshold_value
        ] = self.threshold_value

    def _pre_water_erosion_steps(self):
        self.update_threshold_field()

    def run_one_step(self, step):
        """Advance model **BasicDdSt** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Calculates detachment-limited, threshold-modified erosion by water.

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

    em = BasicDdSt.from_file(infile)
    em.run()


if __name__ == "__main__":
    main()
