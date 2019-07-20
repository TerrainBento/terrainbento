# coding: utf8
# !/usr/env/python
"""terrainbento **BasicChRtTh** model program.

Erosion model program using non-linear diffusion, stream power with stream
powe with a smoothed threshold and spatially varying erodibility based on two
bedrock units, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerSmoothThresholdEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `TaylorNonLinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.taylor_nonlinear_hillslope_flux.html>`_
"""

import numpy as np

from landlab.components import (
    StreamPowerSmoothThresholdEroder,
    TaylorNonLinearDiffuser,
)
from terrainbento.base_class import TwoLithologyErosionModel


class BasicChRtTh(TwoLithologyErosionModel):
    r"""**BasicChRtTh** model program.

    This model program combines :py:class:`BasicCh`, :py:class:`BasicTh` and
    :py:class:`BasicRt` programs by allowing for two lithologies, an "upper"
    layer and a "lower" layer, permitting the use of an smooth erosion
    threshold for each lithology, and using non-linear hillslope transport.
    Given a spatially varying contact zone elevation, :math:`\eta_C(x,y))`,
    model **BasicChRtTh** evolves a topographic surface described by
    :math:`\eta` with the following governing equations:

    .. math::

        \frac{\partial \eta}{\partial t} = -\left[\omega
                    - \omega_c (1 - e^{-\omega /\omega_c}) \right]
                     - \nabla q_h

        \omega = K(\eta, \eta_C) Q^{m} S^{n}

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2,

        \omega_c(\eta, \eta_C ) = w \omega_{c1} + (1 - w) \omega_{c2}

        w = \frac{1}{1+\exp \left( -\frac{(\eta -\eta_C )}{W_c}\right)}

        q_h = -DS \left[ 1 + \left( \frac{S}{S_c} \right)^2
              + \left( \frac{S}{S_c} \right)^4
              + ... \left( \frac{S}{S_c} \right)^{2(N-1)} \right]

    where :math:`Q` is the local stream discharge, :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`W_c` is the contact-zone width, :math:`K_1` and :
    math:`K_2` are the erodabilities of the upper and lower lithologies,
    :math:`\omega_{c1}` and :math:`\omega_{c2}` are the erosion thresholds of
    the upper and lower lithologies, and :math:`D` is the regolith transport
    parameter. :math:`w` is a weight used to calculate the effective
    erodibility :math:`K(\eta, \eta_C)` based on the depth to the contact zone
    and the width of the contact zone. :math:`N` is the number of terms in the
    Taylor Series expansion.

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
    """

    _required_fields = [
        "topographic__elevation",
        "lithology_contact__elevation",
    ]

    def __init__(
        self,
        clock,
        grid,
        water_erosion_rule_upper__threshold=1.0,
        water_erosion_rule_lower__threshold=1.0,
        critical_slope=0.3,
        number_of_taylor_terms=7,
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
        water_erosion_rule_upper__threshold : float, optional.
            Erosion threshold of the upper layer (:math:`\omega_{c1}`). Default
            is 1.
        water_erosion_rule_lower__threshold: float, optional.
            Erosion threshold of the upper layer (:math:`\omega_{c2}`). Default
            is 1.
        critical_slope : float, optional
            Critical slope (:math:`S_c`, unitless). Default is 0.3.
        number_of_taylor_terms : int, optional
            Number of terms in the Taylor Series Expansion (:math:`N`). Default
            is 7.
        **kwargs :
            Keyword arguments to pass to :py:class:`TwoLithologyErosionModel`.
            Importantly these arguments specify the precipitator and the runoff
            generator that control the generation of surface water discharge
            (:math:`Q`).

        Returns
        -------
        BasicChRtTh : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicChRtCh**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicChRtTh
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = constant(grid, "lithology_contact__elevation", value=-10.)

        Construct the model.

        >>> model = BasicChRtTh(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        """
        # Call ErosionModel"s init
        super(BasicChRtTh, self).__init__(clock, grid, **kwargs)

        if float(self.n) != 1.0:
            raise ValueError("Model only supports n equals 1.")

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Save the threshold values for rock and till
        self.rock_thresh = water_erosion_rule_lower__threshold
        self.till_thresh = water_erosion_rule_upper__threshold

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till_with_threshold()

        # Instantiate a StreamPowerSmoothThresholdEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.erody,
            threshold_sp=self.threshold,
            m_sp=self.m,
            n_sp=self.n,
            use_Q="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=self.regolith_transport_parameter,
            slope_crit=critical_slope,
            nterms=number_of_taylor_terms,
        )

    def run_one_step(self, step):
        """Advance model **BasicChRtTh** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Creates rain and runoff, then directs and accumulates flow.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Updates the spatially variable erodibility and threshold values
           based on the relative distance between the topographic surface and
           the lithology contact.

        5. Calculates detachment-limited erosion by water.

        6. Calculates topographic change by non-linear diffusion.

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

        # Update the erodibility and threshold field
        self._update_erodibility_and_threshold_fields()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(step, flooded_nodes=flooded)

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

    chrt = BasicChRtTh.from_file(infile)
    chrt.run()


if __name__ == "__main__":
    main()
