# coding: utf8
# !/usr/env/python
"""terrainbento model **BasicSaVs** program.

Erosion model using depth-dependent linear diffusion with a soil layer, basic
stream power, and discharge proportional to effective drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `DepthDependentDiffuser <http://landlab.readthedocs.io/en/release/_modules/landlab/components/depth_dependent_diffusion/hillslope_depth_dependent_linear_flux.html#DepthDependentDiffuser>`_
    5. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/_modules/landlab/components/weathering/exponential_weathering.html#ExponentialWeatherer>`_
"""

import numpy as np

from landlab.components import (
    DepthDependentDiffuser,
    ExponentialWeatherer,
    StreamPowerEroder,
)
from terrainbento.base_class import ErosionModel

_REQUIRED_FIELDS = ["topographic__elevation", "soil__depth"]


class BasicSaVs(ErosionModel):
    r"""**BasicSaVs** model program.

    Given a spatially varying soil thickness :math:`H` and a spatially varying
    bedrock elevation :math:`\eta_b`, model **BasicSaVs** evolves a topographic
    surface described by :math:`\eta` with the following governing equations:


    .. math::

        \eta = \eta_b + H

        \\frac{\partial H}{\partial t} = P_0 \exp (-H/H_s) - \delta (H) K A_{eff}^{M} S^{N} -\\nabla q_h

        \\frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s) - (1 - \delta (H) ) K A_{eff}^{m} S^{N}

        q_h = -D \left[1-\exp \left( -\\frac{H}{H_0} \\right) \\right] \\nabla \eta

        A_{eff} = A \exp \left( -\\frac{-\\alpha S}{A}\\right)

        \\alpha = \\frac{K_{sat}  H_{init}  dx}{R_m}


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`D` is the regolith transport
    parameter, :math:`H_s` is the sediment production decay depth, :math:`H_0`
    is the sediment transport decay depth, :math:`P_0` is the maximum sediment
    production rate, and :math:`H_0` is the sediment transport decay depth.
    :math:`q_h` is the hillslope sediment flux per unit width.

    :math:`\\alpha` is the saturation area scale used for transforming area into
    effective area :math:`A_{eff}`. It is given as a function of the saturated
    hydraulic conductivity :math:`K_{sat}`, the soil thickness :math:`H_{init}`,
    the grid spacing :math:`dx`, and the recharge rate, :math:`R_m`.

    The **BasicSaVs** program inherits from the terrainbento **ErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+-----------------------------------+
    | Parameter Symbol | Input File Name                   |
    +==================+===================================+
    |:math:`m`         | ``m_sp``                          |
    +------------------+-----------------------------------+
    |:math:`n`         | ``n_sp``                          |
    +------------------+-----------------------------------+
    |:math:`K`         | ``water_erodability``             |
    +------------------+-----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter``  |
    +------------------+-----------------------------------+
    |:math:`K_{sat}`   | ``hydraulic_conductivity``        |
    +------------------+-----------------------------------+
    |:math:`R_m`       | ``recharge_rate``                 |
    +------------------+-----------------------------------+
    |:math:`P_{0}`     | ``soil_production__maximum_rate`` |
    +------------------+-----------------------------------+
    |:math:`H_{s}`     | ``soil_production__decay_depth``  |
    +------------------+-----------------------------------+
    |:math:`H_{0}`     | ``soil_transport_decay_depth``   |
    +------------------+-----------------------------------+

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
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        soil_production__maximum_rate=0.001,
        soil_production__decay_depth=0.5,
        soil_transport_decay_depth=0.5,
        recharge_rate=1.0,
        hydraulic_conductivity=0.1,
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
        BasicSaVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSaVs**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicSaVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")

        Construct the model.

        >>> model = BasicSaVs(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicSaVs, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(_REQUIRED_FIELDS)

        # Get Parameters and convert units if necessary:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability

        soil_thickness = self.grid.at_node["soil__depth"]
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")
        bedrock_elev[:] = self.z - soil_thickness

        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-length parameter
        self.sat_len = (hydraulic_conductivity * self.grid.dx) / (
            recharge_rate
        )

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
            self.grid,
            use_Q="surface_water__discharge",
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
        )

        # Instantiate a DepthDependentDiffuser component
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

    def _calc_effective_drainage_area(self):
        """Calculate and store effective drainage area."""
        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        soil = self.grid.at_node["soil__depth"]
        cores = self.grid.core_nodes
        self.eff_area[cores] = area[cores] * (
            np.exp(-self.sat_len * soil[cores] * slope[cores] / area[cores])
        )

    def run_one_step(self, step):
        """Advance model **BasicVs** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the two erodability by water values.

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
        self.eff_area[flooded] = 0.0

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            self.eroder._K_unit_time.fill(
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(step)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

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

    vssa = BasicSaVs(input_file=infile)
    vssa.run()


if __name__ == "__main__":
    main()
