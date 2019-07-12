# coding: utf8
# !/usr/env/python
"""terrainbento **BasicChSa** model program.

Erosion model program using depth-dependent cubic diffusion with a soil layer,
basic stream power, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/_modules/landlab/components/weathering/exponential_weathering.html#ExponentialWeatherer>`_
    5. `DepthDependentTaylorDiffuser <http://landlab.readthedocs.io/en/release/_modules/landlab/components/depth_dependent_taylor_soil_creep/hillslope_depth_dependent_taylor_flux.html#DepthDependentTaylorDiffuser>`_
"""

import numpy as np

from landlab.components import (
    DepthDependentTaylorDiffuser,
    ExponentialWeatherer,
    FastscapeEroder,
)
from terrainbento.base_class import ErosionModel


class BasicChSa(ErosionModel):
    r"""**BasicChSa** model program.

    This model program combines models :py:class:`BasicCh` and
    :py:class:`BasicSa`. A soil layer is produced by weathering that decays
    exponentially with soil thickness and hillslope transport is soil-depth
    dependent. Given a spatially varying soil thickness :math:`H` and a
    spatially varying bedrock elevation :math:`\eta_b`, model **BasicChSa**
    evolves a topographic surface described by :math:`\eta` with the following
    governing equations:

    .. math::

        \eta = \eta_b + H

        \frac{\partial H}{\partial t} = P_0 \exp (-H/H_s)
                                        - \delta (H) K Q^{m} S^{n}
                                        -\nabla q_h

        \frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s)
                                             - (1 - \delta (H) ) K Q^{m} S^{n}

        q_h = -D S H^* \left[ 1 + \left( \frac{S}{S_c} \right)^2
              + \left( \frac{S}{S_c} \right)^4
              + ... \left( \frac{S}{S_c} \right)^{2(N-1)} \right]

    where :math:`Q` is the local stream discharge, :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`K` is the erodibility by water, :math:`D` is the
    regolith transport parameter, :math:`H_s` is the sediment production decay
    depth, :math:`H_s` is the sediment production decay depth, :math:`P_0` is
    the maximum sediment production rate, and :math:`H_0` is the sediment
    transport decay depth. :math:`q_h` is the hillslope sediment flux per unit
    width. :math:`S_c` is the critical slope parameter and :math:`N` is the
    number of terms in the Taylor Series expansion.

    The function :math:`\delta (H)` is used to indicate that water erosion will
    act on soil where it exists, and on the underlying lithology where soil is
    absent. To achieve this, :math:`\delta (H)` is defined to equal 1 when
    :math:`H > 0` (meaning soil is present), and 0 if :math:`H = 0` (meaning
    the underlying parent material is exposed).

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
        critical_slope=0.3,
        number_of_taylor_terms=11,
        soil_production__maximum_rate=0.001,
        soil_production__decay_depth=0.5,
        soil_transport_decay_depth=0.5,
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
        critical_slope : float, optional
            Critical slope (:math:`S_c`, unitless). Default is 0.3.
        number_of_taylor_terms : int, optional
            Number of terms in the Taylor Series Expansion (:math:`N`). Default
            is 11.
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
        BasicChSa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicChSa**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import constant
        >>> from terrainbento import Clock, BasicChSa
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = constant(grid, "topographic__elevation", value=1.0)
        >>> _ = constant(grid, "soil__depth", value=1.0)

        Construct the model.

        >>> model = BasicChSa(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(10)
        >>> model.model_time
        10.0

        """

        # Call ErosionModel"s init
        super(BasicChSa, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility

        # Create bedrock elevation field
        soil_thickness = self.grid.at_node["soil__depth"]
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")
        bedrock_elev[:] = self.z - soil_thickness

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a weathering component
        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=soil_production__maximum_rate,
            soil_production__decay_depth=soil_production__decay_depth,
        )

        # Instantiate a soil-transport component
        self.diffuser = DepthDependentTaylorDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=critical_slope,
            soil_transport_decay_depth=soil_transport_decay_depth,
            nterms=number_of_taylor_terms,
        )

    def run_one_step(self, step):
        """Advance model **BasicChSa** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Calculates detachment-limited erosion by water.

        5. Produces soil and calculates soil depth with exponential weathering.

        6. Calculates topographic change by depth-dependent nonlinear
           diffusion.

        7. Finalizes the step using the :py:mod:`ErosionModel` base class
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
            self.eroder.K = (
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodibility_adjustment_factor()
            )

        self.eroder.run_one_step(step, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Do some soil creep
        self.diffuser.run_one_step(
            step, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

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

    cdsp = BasicChSa.from_file(infile)
    cdsp.run()


if __name__ == "__main__":
    main()
