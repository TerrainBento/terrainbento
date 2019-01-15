# coding: utf8
# !/usr/env/python
"""terrainbento **BasicRtVs** model program.

Erosion model program using linear diffusion, stream power with spatially
varying erodability based on two bedrock units, and discharge proportional to
effective drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, StreamPowerEroder
from terrainbento.base_class import TwoLithologyErosionModel

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicRtVs(TwoLithologyErosionModel):
    r"""**BasicRtVs** model program.

    **BasicRtVs** is a model program that combines the **BasicRt** and
    **BasicVs** programs by allowing for two lithologies, an "upper" layer and a
    "lower" layer, and using discharge proportional to effective drainage area
    based on variable source area hydrology. Given a spatially varying contact
    zone elevation, :math:`\eta_C(x,y))`, model **BasicRtVs** evolves a
    topographic surface described by :math:`\eta` with the following governing
    equations:


    .. math::

        \\frac{\partial \eta}{\partial t} = - K(\eta,\eta_C) A_{eff}^{m}S^{n} + D\\nabla^2 \eta

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \\frac{1}{1+\exp \left( -\\frac{(\eta -\eta_C )}{W_c}\\right)}

        A_{eff} = A \exp \left( -\\frac{-\\alpha S}{A}\\right)

        \\alpha = \\frac{K_{sat}  H_{init}  dx }{R_m}


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`W_c` is the contact-zone width, :math:`K_1` and :math:`K_2` are the
    erodabilities of the upper and lower lithologies, and :math:`D` is the
    regolith transport parameter. :math:`\\alpha` is the saturation area scale
    used for transforming area into effective area and it is given as a function
    of the saturated hydraulic conductivity :math:`K_{sat}`, the soil thickness
    :math:`H_{init}`, the grid spacing :math:`dx`, and the recharge rate, :math:`R_m`.
    :math:`w` is a weight used to calculate the effective erodability :math:`K(\eta, \eta_C)`
    based on the depth to the contact zone and the width of the contact zone.

    The weight :math:`w` promotes smoothness in the solution of erodability at a
    given point. When the surface elevation is at the contact elevation, the
    erodability is the average of :math:`K_1` and :math:`K_2`; above and below
    the contact, the erodability approaches the value of :math:`K_1` and :math:`K_2`
    at a rate related to the contact zone width. Thus, to make a very sharp
    transition, use a small value for the contact zone width.

    The **BasicRtVs** program inherits from the terrainbento
    **TwoLithologyErosionModel** base class. In addition to the parameters
    required by the base class, models built with this program require the
    following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Name                  |
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
    |:math:`K_{sat}`   | ``hydraulic_conductivity``       |
    +------------------+----------------------------------+
    |:math:`R_m`       | ``recharge_rate``                |
    +------------------+----------------------------------+

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.
    """

    def __init__(
        self,
        clock,
        grid,
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
            :py:class:`~terrainbento.base_class.two_lithology_erosion_model.TwoLithologyErosionModel`.

        Returns
        -------
        BasicRtVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicRtVs**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random, constant
        >>> from terrainbento import Clock, BasicRtVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")
        >>> _ = constant(grid, "lithology_contact__elevation", constant=-10.)

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

        # verify correct fields are present.
        self._verify_fields(_REQUIRED_FIELDS)

        soil_thickness = self.grid.at_node["soil__depth"]

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-area parameter
        self.sat_param = (
            hydraulic_conductivity * soil_thickness * self.grid.dx
        ) / (recharge_rate)

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
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

        $A_{eff} = A \exp ( \alpha S / A) = A R_r$

        where $S$ is downslope-positive steepest gradient, $A$ is drainage
        area, $R_r$ is the runoff ratio, and $\alpha$ is the saturation
        parameter.
        """
        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        cores = self.grid.core_nodes
        self.eff_area[cores] = area[cores] * (
            np.exp(-self.sat_param[cores] * slope[cores] / area[cores])
        )

    def run_one_step(self, step):
        """Advance model **BasicRtVs** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the two erodability by water values.

        4. Updates the spatially variable erodability value based on the
           relative distance between the topographic surface and the lithology
           contact.

        5. Calculates detachment-limited erosion by water.

        6. Calculates topographic change by linear diffusion.

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

        # Update the erodability field
        self._update_erodability_field()

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

    vsrt = BasicRtVs(input_file=infile)
    vsrt.run()


if __name__ == "__main__":
    main()
