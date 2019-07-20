# coding: utf8
# !/usr/env/python
"""terrainbento **BasicRtVs** model program.

Erosion model program using linear diffusion, stream power with spatiall
varying erodibility based on two bedrock units, and discharge proportional to
effective drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import TwoLithologyErosionModel


class BasicRtVs(TwoLithologyErosionModel):
    r"""**BasicRtVs** model program.

    This model program combines the :py:class:`BasicRt` and :py:class:`BasicVs`
    programs by allowing for two lithologies, an "upper" layer and a "lower"
    layer, and using discharge proportional to effective drainage area based on
    variable source area hydrology. Given a spatially varying contact zone
    elevation, :math:`\eta_C(x,y))`, model **BasicRtVs** evolves a topographic
    surface described by :math:`\eta` with the following governing equations:

    .. math::

        \frac{\partial \eta}{\partial t} = - K(\eta,\eta_C) A_{eff}^{m}S^{n}
                                           + D\nabla^2 \eta

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \frac{1}{1+\exp \left( -\frac{(\eta -\eta_C )}{W_c}\right)}

        A_{eff} = A \exp \left( -\frac{-\alpha S}{A}\right)

        \alpha = \frac{K_{sat} dx }{R_m}


    where :math:`Q` is the local stream discharge, :math:`S` is the local
    slope, :math:`m` and :math:`n` are the discharge and slope exponent
    parameters, :math:`W_c` is the contact-zone width, :math:`K_1` and
    :math:`K_2` are the erodabilities of the upper and lower lithologies, and
    :math:`D` is the regolith transport parameter. :math:`\alpha` is the
    saturation area scale used for transforming area into effective area and it
    is given as a function of the saturated hydraulic conductivity
    :math:`K_{sat}`, the soil thickness :math:`H`, the grid spacing :math:`dx`,
    and the recharge rate, :math:`R_m`. :math:`w` is a weight used to calculate
    the effective erodibility :math:`K(\eta, \eta_C)` based on the depth to
    the contact zone and the width of the contact zone.

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

    def __init__(self, clock, grid, hydraulic_conductivity=0.1, **kwargs):
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
        hydraulic_conductivity : float, optional
            Hydraulic conductivity (:math:`K_{sat}`). Default is 0.1.
        **kwargs :
            Keyword arguments to pass to :py:class:`TwoLithologyErosionModel`.
            Importantly these arguments specify the precipitator and the runoff
            generator that control the generation of surface water discharge
            (:math:`Q`).

        Returns
        -------
        BasicRtVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicRtVs**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicRtVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")
        >>> _ = constant(grid, "lithology_contact__elevation", value=-10.)

        Construct the model.

        >>> model = BasicRtVs(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicRtVs, self).__init__(clock, grid, **kwargs)

        # ensure Precipitator and RunoffGenerator are vanilla
        self._ensure_precip_runoff_are_vanilla()

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Get the effective-area parameter
        self._Kdx = hydraulic_conductivity * self.grid.dx

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.erody,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=self.regolith_transport_parameter
        )

    def _calc_effective_drainage_area(self):
        r"""Calculate and store effective drainage area.

        Effective drainage area is defined as:

        .. math::

            A_{eff} = A \exp ( \alpha S / A) = A R_r

        where :math:`S` is downslope-positive steepest gradient, :math:`A` is
        drainage area, :math:`R_r` is the runoff ratio, and :math:`\alpha` is
        the saturation parameter.
        """
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
        """Advance model **BasicRtVs** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

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

        # Update the erodibility field
        self._update_erodibility_field()

        # Do some erosion (but not on the flooded nodes)
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

    vsrt = BasicRtVs.from_file(infile)
    vsrt.run()


if __name__ == "__main__":
    main()
