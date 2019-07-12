# coding: utf8
# !/usr/env/python
"""terrainbento **BasicRt** model program.

Erosion model program using depth-dependent linear diffusion, soil production
by exponential weathering, stream power with spatially varying erodibility based
on two bedrock units, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `DepthDependentDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.depth_dependent_diffusion.html>`_
    5. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/landlab.components.weathering.html>`_
"""

import numpy as np

from landlab.components import (
    DepthDependentDiffuser,
    ExponentialWeatherer,
    FastscapeEroder,
)
from terrainbento.base_class import TwoLithologyErosionModel


class BasicRtSa(TwoLithologyErosionModel):
    r"""**BasicRtSa** model program.

    This model program combines the :py:class:`BasicRt` and :py:class:`BasicSa`
    programs by allowing for two lithologies, an "upper" layer and a "lower"
    layer and explicitly resolving a soil layer. This soil layer is produced by
    weathering that decays exponentially with soil thickness and hillslope
    transport is soil-depth dependent. Given a spatially varying contact zone
    elevation, :math:`\eta_C(x,y))`, a spatially varying soil thickness
    :math:`H` and a spatially varying bedrock elevation :math:`\eta_b`,
    model **BasicRtSa** evolves a topographic surface described by :math:`\eta`
    with the following governing equations:

    .. math::

        \eta = \eta_b + H

        \frac{\partial H}{\partial t} = P_0 \exp (-H/H_s)
                                        - \delta (H) K Q^{m} S^{n}
                                        - \nabla q_h

        \frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s)
                                             - (1 - \delta (H) ) K Q^{m} S^{n}

        q_h = -D H^* \left[1-\exp \left( -\frac{H}{H_0} \right) \right] \nabla \eta

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \frac{1}{1+\exp \left( -\frac{(\eta -\eta_C )}{W_c}\right)}

    where :math:`Q` is the local stream discharge, :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`W_c` is the contact-zone width, :math:`K_1` and
    :math:`K_2` are the erodabilities of the upper and lower lithologies, and
    :math:`D` is the regolith transport parameter. :math:`w` is a weight used
    to calculate the effective erodibility :math:`K(\eta, \eta_C)` based on the
    depth to the contact zone and the width of the contact zone. :math:`H_s` is
    the sediment production decay depth, :math:`H_0` is the sediment transport
    decay depth, :math:`P_0` is the maximum sediment production rate, and
    :math:`H_0` is the sediment transport decay depth. :math:`q_h` is the
    hillslope sediment flux per unit width.

    The function :math:`\delta (H)` is used to indicate that water erosion will
    act on soil where it exists, and on the underlying lithology where soil is
    absent. To achieve this, :math:`\delta (H)` is defined to equal 1 when
    :math:`H > 0` (meaning soil is present), and 0 if :math:`H = 0` (meaning
    the underlying parent material is exposed).

    The weight :math:`w` promotes smoothness in the solution of erodibility at
    a given point. When the surface elevation is at the contact elevation, the
    erodibility is the average of :math:`K_1` and :math:`K_2`; above and below
    the contact, the erodibility approaches the value of :math:`K_1` and
    :math:`K_2` at a rate related to the contact zone width. Thus, to make a
    very sharp transition, use a small value for the contact zone width.

    Refer to
    `Barnhart et al. (2019) <https://doi.org/10.5194/gmd-12-1267-2019>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
        - ``lithology_contact__elevation``
        - ``soil__depth``
    """

    _required_fields = [
        "topographic__elevation",
        "lithology_contact__elevation",
        "soil__depth",
    ]

    def __init__(
        self,
        clock,
        grid,
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
        water_erodibility_upper : float, optional
            Water erodibility of the upper layer (:math:`K_{1}`). Default is
            0.001.
        water_erodibility_lower : float, optional
            Water erodibility of the upper layer (:math:`K_{2}`). Default is
            0.0001.
        contact_zone__width : float, optional
            Thickness of the contact zone (:math:`W_c`). Default is 1.
        regolith_transport_parameter : float, optional
            Regolith transport efficiency (:math:`D`). Default is 0.1.
        soil_production__maximum_rate : float, optional
            Maximum rate of soil production (:math:`P_{0}`). Default is 0.001.
        soil_production__decay_depth : float, optional
            Decay depth for soil production (:math:`H_{s}`). Default is 0.5.
        soil_transport_decay_depth : float, optional
            Decay depth for soil transport (:math:`H_{0}`). Default is 0.5.
        **kwargs :
            Keyword arguments to pass to :py:class:`TwoLithologyErosionModel`.
            Importantly these arguments specify the precipitator and the runoff
            generator that control the generation of surface water discharge
            (:math:`Q`).

        Returns
        -------
        BasicRtSa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicRtSa**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicRtSa
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")
        >>> _ = constant(grid, "lithology_contact__elevation", value=-10.)

        Construct the model.

        >>> model = BasicRtSa(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicRtSa, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.erody,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        soil_thickness = self.grid.at_node["soil__depth"]
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")
        bedrock_elev[:] = self.z - soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=self.regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=soil_production__maximum_rate,
            soil_production__decay_depth=soil_production__decay_depth,
        )

    def run_one_step(self, step):
        """Advance model **BasicRtSa** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Updates the spatially variable erodibility value based on the
           relative distance between the topographic surface and the lithology
           contact.

        5. Calculates detachment-limited erosion by water.

        6. Calculates topographic change by linear diffusion.

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

        # Update the erodibility field
        self._update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(
            step, flooded_nodes=flooded, K_if_used=self.erody
        )

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


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    sart = BasicRtSa.from_file(infile)
    sart.run()


if __name__ == "__main__":
    main()
