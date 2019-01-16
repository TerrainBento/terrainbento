# coding: utf8
# !/usr/env/python
"""terrainbento **BasicDdHy** model program.

Erosion model program using linear diffusion, stream-power-driven sediment
erosion and mass conservation with a smoothed threshold that varies with
incision depth, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicDdHy(ErosionModel):
    r"""**BasicDdHy** model program.

    **BasicDdHy** is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:


    .. math::

        \\\frac{\partial \eta}{\partial t} = -\left(KA^{m}S^{n} - \omega_{ct}\left(1-e^{-KA^{m}S^{n}/\omega_{ct}}\\right)\\right) + \\\frac{V\\\frac{Q_s}{Q}}{\left(1-\phi\\right)} + D\nabla^2 \eta


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`\omega_{ct}` is the critical
    stream power needed for erosion to occur, :math:`V` is effective sediment
    settling velocity, :math:`Q_s` is volumetric sediment flux, :math:`Q` is
    volumetric water discharge, :math:`\phi` is sediment porosity,  and
    :math:`D` is the regolith transport efficiency.

    :math:`\omega_{ct}` may change through time as it increases with cumulative
    incision depth:

    .. math::

        \omega_{ct}\left(x,y,t\\right) = \mathrm{max}\left(\omega_c + b D_I\left(x, y, t\\right), \omega_c \\right)

    where :math:`\omega_c` is the threshold when no incision has taken place,
    :math:`b` is the rate at which the threshold increases with incision depth,
    and :math:`D_I` is the cumulative incision depth at location
    :math:`\left(x,y\\right)` and time :math:`t`.

    The **BasicDdHy** program inherits from the terrainbento **ErosionModel**
    base class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +--------------------+-------------------------------------------------+
    | Parameter Symbol   | Input File Parameter Name                       |
    +====================+=================================================+
    |:math:`m`           | ``m_sp``                                        |
    +--------------------+-------------------------------------------------+
    |:math:`n`           | ``n_sp``                                        |
    +--------------------+-------------------------------------------------+
    |:math:`K`           | ``water_erodability``                           |
    +--------------------+-------------------------------------------------+
    |:math:`D`           | ``regolith_transport_parameter``                |
    +--------------------+-------------------------------------------------+
    |:math:`V`           | ``settling_velocity``                           |
    +--------------------+-------------------------------------------------+
    |:math:`F_f`         | ``fraction_fines``                              |
    +--------------------+-------------------------------------------------+
    |:math:`\phi`        | ``sediment_porosity``                           |
    +--------------------+-------------------------------------------------+
    |:math:`\omega_{c}`  | ``water_erosion_rule__threshold``               |
    +--------------------+-------------------------------------------------+
    |:math:`b`           | ``water_erosion_rule__thresh_depth_derivative`` |
    +--------------------+-------------------------------------------------+

    Refer to
    `Barnhart et al. (2019) <https://www.geosci-model-dev-discuss.net/gmd-2018-204/>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        water_erosion_rule__threshold=0.01,
        water_erosion_rule__thresh_depth_derivative=0.,
        settling_velocity=0.001,
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
            :py:class:`~terrainbento.base_class.erosion_model.ErosionModel`.

        Returns
        -------
        BasicDdHy : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDdHy**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicDdHy
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicDdHy(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        """
        # Call ErosionModel"s init
        super(BasicDdHy, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Get Parameters and convert units if necessary:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability
        self.sp_crit = water_erosion_rule__threshold

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros(
            "node", "water_erosion_rule__threshold"
        )
        self.threshold[:] = self.sp_crit  # starting value

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            F_f=fraction_fines,
            phi=sediment_porosity,
            v_s=settling_velocity,
            m_sp=self.m,
            n_sp=self.n,
            sp_crit="water_erosion_rule__threshold",
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = (
            water_erosion_rule__thresh_depth_derivative
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, step):
        """Advance model **BasicDdHy** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates threshold-modified erosion and deposition by water.

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

        # Calculate cumulative erosion and update threshold
        cum_ero = self.grid.at_node["cumulative_elevation_change"]
        cum_ero[:] = (
            self.z - self.grid.at_node["initial_topographic__elevation"]
        )
        self.threshold[:] = self.sp_crit - (
            self.thresh_change_per_depth * cum_ero
        )
        self.threshold[self.threshold < self.sp_crit] = self.sp_crit

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

    em = BasicDdHy(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
