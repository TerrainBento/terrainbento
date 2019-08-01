# coding: utf8
# !/usr/env/python
"""terrainbento model **BasicThVs** program.

Erosion model program using linear diffusion, stream-power-driven sediment
erosion and mass conservation, and discharge proportional to effective drainage
area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicHyVs(ErosionModel):
    r"""**BasicHyVs** model program.

    This model program combines :py:class:`BasicHy` and :py:class:`BasicVs` to
    evolves a topographic surface described by :math:`\eta` with the following
    governing equations:

    .. math::

        \frac{\partial \eta}{\partial t} = -\left(KQ(A)^{m}S^{n}
              - \omega_c\left(1-e^{-KQ(A)^{m}S^{n}/\omega_c}\right)\right)
              + \frac{V\frac{Q_s}{Q(A)}}{\left(1-\phi\right)}
              + D\nabla^2 \eta

        Q_s = \int_0^A \left(KQ(A)^{m}S^{n} - \frac{V Q_s}{Q(A)} \right) dA

        Q = A \exp \left( -\frac{-\alpha S}{A}\right)

        \alpha = \frac{K_{sat} H dx}{R_m}

    where :math:`Q` is the local stream discharge, :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`K` is the erodibility by water, :math:`\omega_c` is the
    critical stream power needed for erosion to occur, :math:`V` is effective
    sediment settling velocity, :math:`Q_s` is volumetric sediment flux,
    :math:`\phi` is sediment porosity, and :math:`D` is the regolith transport
    efficiency.

    :math:`\alpha` is the saturation area scale used for transforming area into
    effective area :math:`A_{eff}` (used as discharge). It is given as a
    function of the saturated hydraulic conductivity :math:`K_{sat}`, the soil
    thickness :math:`H`, the grid spacing :math:`dx`, and the recharge rate,
    :math:`R_m`.

    Refer to
    `Barnhart et al. (2019) <https://doi.org/10.5194/gmd-12-1267-2019>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
        - ``soil__depth``
    """

    _required_fields = ["topographic__elevation", "soil__depth"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodibility=0.0001,
        regolith_transport_parameter=0.1,
        settling_velocity=0.001,
        sediment_porosity=0.3,
        fraction_fines=0.5,
        hydraulic_conductivity=0.1,
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
            `ErosionDeposition <https://landlab.readthedocs.io/en/latest/landlab.components.erosion_deposition.html>`__
            component. Default is "basic".
        hydraulic_conductivity : float, optional
            Hydraulic conductivity (:math:`K_{sat}`). Default is 0.1.
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        BasicHyVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHy**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicHyVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")

        Construct the model.

        >>> model = BasicHyVs(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel"s init
        super(BasicHyVs, self).__init__(clock, grid, **kwargs)

        # ensure Precipitator and RunoffGenerator are vanilla
        self._ensure_precip_runoff_are_vanilla(vsa_precip=True)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility

        # Get the effective-area parameter
        self._Kdx = hydraulic_conductivity * self.grid.dx

        # Instantiate a SPACE component
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

    def _calc_effective_drainage_area(self):
        """Calculate and store effective drainage area."""

        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        cores = self.grid.core_nodes

        sat_param = (
            self._Kdx
            * self.grid.at_node["soil__depth"]
            / self.grid.at_node["rainfall__flux"]
        )

        eff_area = area[cores] * (
            np.exp(-sat_param[cores] * slope[cores] / area[cores])
        )

        self.grid.at_node["surface_water__discharge"][cores] = eff_area

    def run_one_step(self, step):
        """Advance model **BasicVs** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

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

        # Update effective runoff ratio
        self._calc_effective_drainage_area()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Zero out effective area in flooded nodes
        self.grid.at_node["surface_water__discharge"][flooded] = 0.0

        # Do some erosion
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

    my_model = BasicHyVs.from_file(infile)
    my_model.run()


if __name__ == "__main__":
    main()
