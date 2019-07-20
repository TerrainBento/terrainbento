# coding: utf8
# !/usr/env/python
"""terrainbento model **BasicHySa** program.

Erosion model program using exponential weathering, soil-depth-dependent linear
diffusion, stream-power-driven sediment erosion, mass conservation, and bedrock
erosion, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `Space <http://landlab.readthedocs.io/en/release/landlab.components.space.html>`_
    4. `DepthDependentDiffuser <http://http://landlab.readthedocs.io/en/release/landlab.components.depth_dependent_diffusion.html>`_
    5. `ExponentialWeatherer <http://http://landlab.readthedocs.io/en/release/landlab.components.weathering.html>`_
"""

import numpy as np

from landlab.components import (
    DepthDependentDiffuser,
    ExponentialWeatherer,
    Space,
)
from terrainbento.base_class import ErosionModel


class BasicHySa(ErosionModel):
    r"""**BasicHySa** program.

    This model program combines :py:class:`BasicHy` and :py:class:`BasicSa` to
    evolve a topographic surface described by :math:`\eta` with the following
    governing equation:

    .. math::

        \eta = \eta_b + H

        \frac{\partial H}{\partial t} = P_0 \exp (-H/H_s)
                          + \frac{V_s Q_s}{Q(A)\left(1 - \phi \right)}
                          - K_s Q(A)^{m}S^{n} (1 - e^{-H/H_*})
                          -\nabla q_h

        \frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s)
                                             - K_r Q(A)^{m}S^{n} e^{-H/H_*}

        Q_s = \int_0^A \left(K_s Q(A)^{m}S^{n} (1-e^{-H/H_*})
              + K_r (1-F_f) Q(A)^{m}S^{n} e^{-H/H_*}
              - \frac{V_s Q_s}{Q(A)}\right) dA

    where :math:`\eta_b` is the bedrock elevation, :math:`H` is the soil depth,
    :math:`P_0` is the maximum soil production rate, :math:`H_s` is the soil
    production decay depth, :math:`V_s` is effective sediment settling
    velocity, :math:`Q_s` is volumetric fluvial sediment flux, :math:`A` is the
    local drainage area, :math:`Q`, is the local discharge, :math:`S` is the
    local slope, :math:`\phi` is sediment porosity, :math:`F_f` is the fraction
    of fine sediment, :math:`K_r` and :math:`K_s` are rock and sediment
    erodibility respectively, :math:`m` and :math:`n` are the discharge and
    slope exponent parameters, :math:`H_*` is the bedrock roughness length
    scale, and :math:`r` is a runoff rate. Hillslope sediment flux per unit
    width :math:`q_h` is given by:

    .. math::

        q_h = -D H^* \left[1-\exp \left( -\frac{H}{H_0} \right) \right]
              \nabla \eta.

    where :math:`D` is soil diffusivity and :math:`H_0` is the soil transport
    depth scale.

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
        water_erodibility_sediment=0.001,
        water_erodibility_rock=0.0001,
        regolith_transport_parameter=0.1,
        settling_velocity=0.001,
        sediment_porosity=0.3,
        fraction_fines=0.5,
        roughness__length_scale=0.5,
        solver="basic",
        soil_production__maximum_rate=0.001,
        soil_production__decay_depth=0.5,
        soil_transport_decay_depth=0.5,
        sp_crit_br=0,
        sp_crit_sed=0,
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
            Normalized settling velocity of entrained sediment (:math:`V_s`).
            Default is 0.001.
        sediment_porosity : float, optional
            Sediment porosity (:math:`\phi`). Default is 0.3.
        fraction_fines : float, optional
            Fraction of fine sediment that is permanently detached
            (:math:`F_f`). Default is 0.5.
        roughness__length_scale : float, optional
            Bedrock roughness length scale. Default is 0.5.
        solver : str, optional
            Solver option to pass to the Landlab
            `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.space.html>`_
            component. Default is "basic".
        soil_production__maximum_rate : float, optional
            Maximum rate of soil production (:math:`P_{0}`). Default is 0.001.
        soil_production__decay_depth : float, optional
            Decay depth for soil production (:math:`H_{s}`). Default is 0.5.
        soil_transport_decay_depth : float, optional
            Decay depth for soil transport (:math:`H_{0}`). Default is 0.5.
        **kwargs :
            Keyword arguments to pass to :py:class:`ErosionModel`. Importantly
            these arguments specify the precipitator and the runoff generator
            that control the generation of surface water discharge (:math:`Q`).

        Returns
        -------
        BasicHySa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHySa**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicHySa
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")

        Construct the model.

        >>> model = BasicHySa(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicHySa, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        soil_thickness = self.grid.at_node["soil__depth"]
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")
        bedrock_elev[:] = self.z - soil_thickness

        self.m = m_sp
        self.n = n_sp
        self.K_br = water_erodibility_rock
        self.K_sed = water_erodibility_sediment

        # Instantiate a SPACE component
        self.eroder = Space(
            self.grid,
            K_sed=self.K_sed,
            K_br=self.K_br,
            sp_crit_br=sp_crit_br,
            sp_crit_sed=sp_crit_sed,
            F_f=fraction_fines,
            phi=sediment_porosity,
            H_star=roughness__length_scale,
            v_s=settling_velocity,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=soil_production__maximum_rate,
            soil_production__decay_depth=soil_production__decay_depth,
        )

        self.grid.at_node["soil__depth"][:] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["bedrock__elevation"]
        )

    def run_one_step(self, step):
        """Advance model **BasicHySa** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Calculates erosion and deposition by water.

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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            erode_factor = self.boundary_handlers[
                "PrecipChanger"
            ].get_erodibility_adjustment_factor()
            self.eroder.K_sed = self.K_sed * erode_factor
            self.eroder.K_br = self.K_br * erode_factor

        self.eroder.run_one_step(step, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
        self.diffuser.run_one_step(step)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(step)

        # Check stability
        self.check_stability()

    def check_stability(self):
        """Check model stability and exit if unstable."""
        fields = self.grid.at_node.keys()
        for f in fields:
            if np.any(np.isnan(self.grid.at_node[f])) or np.any(
                np.isinf(self.grid.at_node[f])
            ):
                raise SystemExit(
                    "terrainbento ModelHySa: Model became unstable"
                )


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    hysa = BasicHySa.from_file(infile)
    hysa.run()


if __name__ == "__main__":
    main()
