# coding: utf8
# !/usr/env/python
"""terrainbento **BasicDd** model program.

Erosion model program using linear diffusion, stream power with a smoothed
threshold that varies with incision depth, and discharge proportional to
drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerSmoothThresholdEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, StreamPowerSmoothThresholdEroder
from terrainbento.base_class import ErosionModel


class BasicDd(ErosionModel):
    r"""**BasicDd** model program.

    This model program evolves a topographic surface, :math:`\eta`, with the
    following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -\left(KQ^{m}S^{n}
                - \omega_{ct}\left(1-e^{-KQ^{m}S^{n}/\omega_{ct}}\right)\right)
                + D\nabla^2 \eta

    where :math:`Q` is the local stream discharge and :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`K` is the erodibility by water, :math:`D` is the
    regolith transport efficiency, and :math:`\omega_{ct}` is the critical
    stream power needed for erosion to occur. :math:`\omega_{ct}` changes
    through time as it increases with cumulative incision depth:

    .. math::

        \omega_{ct}\left(x,y,t\right) = \mathrm{max}\left(\omega_c +
                                   b D_I\left(x, y, t\right), \omega_c \right)

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
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        BasicDd : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDd**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicDd
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicDd(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        """
        # Call ErosionModel"s init
        super(BasicDd, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Get Parameters and convert units if necessary:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility

        if float(self.n) != 1.0:
            raise ValueError("Model only supports n equals 1.")

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        self.threshold_value = water_erosion_rule__threshold

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
            threshold_sp=self.threshold,
            use_Q="surface_water__discharge",
        )

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = (
            water_erosion_rule__thresh_depth_derivative
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def update_erosion_threshold_values(self):
        r"""Update the erosion threshold at each node based on cumulative
        incision so far using:

        .. math::

            \omega_{ct}\left(x,y,t\right) = \mathrm{max}\left(\omega_c + \\
            b D_I\left(x, y, t\right), \omega_c \right)

        where :math:`\omega_c` is the threshold when no incision has taken
        place, :math:`b` is the rate at which the threshold increases with
        incision depth, and :math:`D_I` is the cumulative incision depth at
        location :math:`\left(x,y\right)` and time :math:`t`.
        """

        # Set the erosion threshold.
        #
        # Note that a minus sign is used because cum ero depth is negative for
        # erosion, positive for deposition.
        # The second line handles the case where there is growth, in which case
        # we want the threshold to stay at its initial value rather than
        # getting smaller.
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

    def run_one_step(self, step):
        """Advance model **BasicDd** for one time-step of duration step.

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

        # Calculate the new threshold values given cumulative erosion
        self.update_erosion_threshold_values()

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            self.eroder.K = (
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodibility_adjustment_factor()
            )
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

    ldsp = BasicDd.from_file(infile)
    ldsp.run()


if __name__ == "__main__":
    main()
