# coding: utf8
# !/usr/env/python
"""terrainbento **BasicHyRt** model program.

Erosion model program using linear diffusion, stream-power-driven sediment
erosion and mass conservation with spatially varying erodibility based on two
bedrock units, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import TwoLithologyErosionModel


class BasicHyRt(TwoLithologyErosionModel):
    r"""**BasicHyRt** model program.

    This model program combines the :py:class:`BasicRt` and :py:class:`BasicHy`
    programs by allowing for two lithologies, an "upper" layer and a "lower"
    layer, stream-power-driven sediment erosion and mass conservation. Given a
    spatially varying contact zone elevation, :math:`\eta_C(x,y))`, model
    **BasicHyRt** evolves a topographic surface described by :math:`\eta` with
    the following governing equations:

    .. math::

        \frac{\partial \eta}{\partial t} = \frac{V Q_s}{Q}
                                           - K Q^{m}S^{n}
                                           + D\nabla^2 \eta

        Q_s = \int_0^A \left((1-F_f)KQ(A)^{m}S^{n}
                             - \frac{V Q_s}{Q(A)} \right) dA

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \frac{1}{1+\exp \left( -\frac{(\eta -\eta_C )}{W_c}\right)}

    where :math:`Q` is the local stream discharge, :math:`A` is the local
    upstream drainage area, :math:`S` is the local slope, :math:`m` and
    :math:`n` are the discharge and slope exponen parameters, :math:`W_c` is
    the contact-zone width, :math:`K_1` and :math:`K_2` are the erodabilities
    of the upper and lower lithologies, and :math:`D` is the regolith transport
    parameter. :math:`Q_s` is the volumetric sediment discharge, and
    :math:`V` is the effective settling velocity of the sediment. :math:`w` is
    a weight used to calculate the effective erodibility
    :math:`K(\eta, \eta_C)` based on the depth to the contact zone and the
    width of the contact zone.

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
        solver="basic",
        settling_velocity=0.001,
        sediment_porosity=0.3,
        fraction_fines=0.5,
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
        **kwargs :
            Keyword arguments to pass to :py:class:`TwoLithologyErosionModel`.
            Importantly these arguments specify the precipitator and the runoff
            generator that control the generation of surface water discharge
            (:math:`Q`).

        Returns
        -------
        BasicHyRt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHyRt**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicHyRt
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = constant(grid, "lithology_contact__elevation", value=-10.)

        Construct the model.

        >>> model = BasicHyRt(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        """
        # Call ErosionModel"s init
        super(BasicHyRt, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Save the threshold values for rock and till
        self.rock_thresh = 0.0
        self.till_thresh = 0.0

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till_with_threshold()

        # Instantiate an ErosionDeposition ("hybrid") component
        self.eroder = ErosionDeposition(
            self.grid,
            K="substrate__erodibility",
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
            self.grid, linear_diffusivity=self.regolith_transport_parameter
        )

    def run_one_step(self, step):
        """Advance model **BasicHyRt** for one time-step of duration step.

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

        # Update the erodibility and threshold field
        self._update_erodibility_and_threshold_fields()

        # Do some erosion (but not on the flooded nodes)
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

    thrt = BasicHyRt.from_file(infile)
    thrt.run()


if __name__ == "__main__":
    main()
