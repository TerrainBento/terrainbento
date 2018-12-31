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

_REQUIRED_FIELDS = ["topographic__elevation", "lithology_contact__elevation"]


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

    *Specifying the Lithology Contact*

    In all two-lithology models the spatially variable elevation of the contact
    elevation must be given as the file path to an ESRII ASCII format file using
    the parameter ``lithology_contact_elevation__file_name``. If topography was
    created using an input DEM, then the shape of the field contained in the
    file must be the same as the input DEM. If synthetic topography is used then
    the shape of the field must be ``number_of_node_rows-2`` by
    ``number_of_node_columns-2``. This is because the read-in DEM will be padded
    by a halo of size 1.

    *Reference Frame Considerations*

    Note that the developers had to make a decision about how to represent the
    contact. We could represent the contact between two layers either as a depth
    below present land surface, or as an altitude. Using a depth would allow for
    vertical motion, because for a fixed surface, the depth remains constant
    while the altitude changes. But the depth must be updated every time the
    surface is eroded or aggrades. Using an altitude avoids having to update the
    contact position every time the surface erodes or aggrades, but any tectonic
    motion would need to be applied to the contact position as well. We chose to
    use the altitude approach because this model was originally written for an
    application with lots of erosion expected but no tectonics.

    If implementing tectonics is desired, consider using either the
    **SingleNodeBaselevelHandler** or the **NotCoreNodeBaselevelHandler** which
    modify both the ``topographic__elevation`` and the ``bedrock__elevation``
    fields.
    """

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        **kwargs
    ):
        """
        Parameters
        ----------


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
        >>> from landlab.values import random
        >>> from terrainbento import Clock, Basic
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")

        Construct the model.

        >>> model = Basic(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        >>> params = {"model_grid": "RasterModelGrid",
        ...           "clock": {"step": 1,
        ...                     "output_interval": 2.,
        ...                     "stop": 200.},
        ...           "number_of_node_rows" : 6,
        ...           "number_of_node_columns" : 9,
        ...           "node_spacing" : 10.0,
        ...           "regolith_transport_parameter": 0.001,
        ...           "water_erodability_lower": 0.001,
        ...           "water_erodability_upper": 0.01,
        ...           "contact_zone__width": 1.0,
        ...           "lithology_contact_elevation__file_name": "tests/data/example_contact_elevation.asc",
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0,
        ...           "recharge_rate": 0.5,
        ...           "hydraulic_conductivity": 0.1}

        Construct the model.

        >>> model = BasicRtVs(params=params)

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

        recharge_rate = (self._length_factor) * recharge_rate
        soil_thickness = self.grid.at_node["soil__depth"]
        K_hydraulic_conductivity = (
            self._length_factor
        ) * hydraulic_conductivity
        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-area parameter
        self.sat_param = (
            K_hydraulic_conductivity * soil_thickness * self.grid.dx
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
