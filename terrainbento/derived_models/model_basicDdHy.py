# coding: utf8
# !/usr/env/python
"""terrainbento **BasicDdHy** model program.

Erosion model program using linear diffusion, stream-power-driven sediment
erosion and mass conservation with a smoothed threshold that varies with
incision depth, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <https://landlab.readthedocs.io/en/master/reference/components/flow_accum.html>`_
    2. `DepressionFinderAndRouter <https://landlab.readthedocs.io/en/master/reference/components/flow_routing.html>`_ (optional)
    3. `ErosionDeposition <https://landlab.readthedocs.io/en/master/reference/components/erosion_deposition.html>`_
    4. `LinearDiffuser <https://landlab.readthedocs.io/en/master/reference/components/diffusion.html>`_
"""

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicDdHy(ErosionModel):
    r"""**BasicDdHy** model program.

    This model program combines models :py:class:`BasicDd` and
    :py:class:`BasicHy`. It evolves a topographic surface, :math:`\eta`, with
    the following governing equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -\left(KQ(A)^{m}S^{n}
            - \omega_{ct}\left(1-e^{-KQ^{m}S^{n}/\omega_{ct}}\right)\right)
            + \frac{V\frac{Q_s}{Q(A)}}{\left(1-\phi\right)}
            + D\nabla^2 \eta

        Q_s = \int_0^A \left((1-F_f)[\omega
              - \omega_c (1 - e^{-\omega / \omega_c})]
              - \frac{V Q_s}{Q(A)} \right) dA

        \omega = KQ(A)^{m}S^{n}

    where :math:`Q` is the local stream discharge, :math:`A` is the local
    upstream drainage area, :math:`S` is the local slope, :math:`m` and
    :math:`n` are the discharge and slope exponent parameters, :math:`K` is the
    erodibility by water, :math:`\omega_{ct}` is the critical stream power
    needed for erosion to occur, :math:`V` is effective sediment settling
    velocity, :math:`Q_s` is volumetric sediment flux, :math:`\phi` is sediment
    porosity, and :math:`D` is the regolith transport efficiency.

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
            (:math:`\omega_c`). Default is 1.0.
        water_erosion_rule__thresh_depth_derivative : float, optional
            Rate of increase of water erosion threshold as increased incision
            occurs (:math:`b`). Default is 0.0.
        settling_velocity : float, optional
            Settling velocity of entrained sediment (:math:`V`). Default
            is 0.001.
        sediment_porosity : float, optional
            Sediment porosity (:math:`\phi`). Default is 0.3.
        fraction_fines : float, optional
            Fraction of fine sediment that is permanently detached
            (:math:`F_f`). Default is 0.5.
        solver : str, optional
            Solver option to pass to the Landlab
            `ErosionDeposition <https://landlab.readthedocs.io/en/master/reference/components/erosion_deposition.html>`__
            component. Default is "basic".
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        BasicDdHy : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDdHy**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

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
        self.K = water_erodibility
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
            erode_flooded_nodes=self._erode_flooded_nodes,
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

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Calculates threshold-modified erosion and deposition by water.

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
                ].get_erodibility_adjustment_factor()
            )
        self.eroder.run_one_step(step)

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

    em = BasicDdHy.from_file(infile)
    em.run()


if __name__ == "__main__":
    main()
