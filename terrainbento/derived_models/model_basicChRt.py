# coding: utf8
# !/usr/env/python
"""terrainbento **BasicChRt** model program.

Erosion model program using non-linear diffusion, stream power with spatially
varying erodability based on two bedrock units, and discharge proportional to
drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `TaylorNonLinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.taylor_nonlinear_hillslope_flux.html>`_
"""

import numpy as np

from landlab.components import FastscapeEroder, TaylorNonLinearDiffuser
from terrainbento.base_class import TwoLithologyErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicChRt(TwoLithologyErosionModel):
    r"""**BasicChRt** model program.

    **BasicChRt** is a model program that combines the **BasicRt** and
    **BasicCh** programs by allowing for two lithologies, an "upper" layer and a
    "lower" layer, and non-linear hillslope sediment transport. Given a
    spatially varying contact zone elevation, :math:`\eta_C(x,y))`, model
    **BasicChRt** evolves a topographic surface described by :math:`\eta` with
    the following governing equations:


    .. math::

        \\frac{\partial \eta}{\partial t} = - K(\eta,\eta_C) A^{m}S^{n} - \\nabla q_h

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \\frac{1}{1+\exp \left( -\\frac{(\eta -\eta_C )}{W_c}\\right)}

        q_h = -DS \left[ 1 + \left( \\frac{S}{S_c} \\right)^2 +  \left( \\frac{S}{S_c} \\right)^4 + ... \left( \\frac{S}{S_c} \\right)^{2(N-1)} \\right]

    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`W_c` is the contact-zone width, :math:`K_1` and :math:`K_2` are the
    erodabilities of the upper and lower lithologies, and :math:`D` is the
    regolith transport parameter. :math:`S_c` is the critical slope parameter
    and :math:`N` is the number of terms in the Taylor Series expansion. :math:`N`
    is set at a default value of 7 but can be modified by a user. :math:`w` is a
    weight used to calculate the effective erodability :math:`K(\eta, \eta_C)`
    based on the depth to the contact zone and the width of the contact zone.

    The weight :math:`w` promotes smoothness in the solution of erodability at a
    given point. When the surface elevation is at the contact elevation, the
    erodability is the average of :math:`K_1` and :math:`K_2`; above and below
    the contact, the erodability approaches the value of :math:`K_1` and :math:`K_2`
    at a rate related to the contact zone width. Thus, to make a very sharp
    transition, use a small value for the contact zone width.

    The **BasicChRt** program inherits from the terrainbento
    **TwoLithologyErosionModel** base class. In addition to the parameters
    required by the base class, models built with this program require the
    following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K_{1}`     | ``water_erodability_upper``      |
    +------------------+----------------------------------+
    |:math:`K_{2}`     | ``water_erodability_lower``      |
    +------------------+----------------------------------+
    |:math:`W_{c}`     | ``contact_zone__width``          |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`S_c`       | ``critical_slope``               |
    +------------------+----------------------------------+
    |:math:`N`         | ``number_of_taylor_terms``       |
    +------------------+----------------------------------+

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.

    """

    def __init__(
        self,
        clock,
        grid,
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

        **kwargs :
            Keyword arguments to pass to
            :py:class:`~terrainbento.base_class.two_lithology_erosion_model.TwoLithologyErosionModel`.

        Returns
        -------
        BasicChRt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicChRt**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicChRt
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = constant(grid, "lithology_contact__elevation", constant=-10.)

        Construct the model.

        >>> model = BasicChRt(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        """
        # Call ErosionModel"s init
        super(BasicChRt, self).__init__(clock, grid, **kwargs)

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            m_sp=self.m,
            n_sp=self.n,
            K_sp=self.erody,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=self.regolith_transport_parameter,
            slope_crit=critical_slope,
            nterms=number_of_taylor_terms,
        )

    def run_one_step(self, step):
        """Advance model **BasicChRt** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the two erodability by water values.

        4. Updates the spatially variable erodability value based on the
           relative distance between the topographic surface and the lithology
           contact.

        5. Calculates detachment-limited erosion by water.

        6. Calculates topographic change by non-linear diffusion.

        7. Finalizes the step using the **ErosionModel** base class function
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

        # Update the erodability field
        self._update_erodability_field()

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(
            step, flooded_nodes=flooded, K_if_used=self.erody
        )

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

    chrt = BasicChRt(input_file=infile)
    chrt.run()


if __name__ == "__main__":
    main()
