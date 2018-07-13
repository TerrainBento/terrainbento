# coding: utf8
#! /usr/env/python
"""terrainbento **BasicRt** model program.

Erosion model program using depth-dependent linear diffusion, soil production
by exponential weathering, stream power with spatially varying erodability based
on two bedrock units, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `DepthDependentDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.depth_dependent_diffusion.html>`_
    5. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/landlab.components.weathering.html>`_
"""

import sys
import numpy as np

from landlab.components import (
    FastscapeEroder,
    DepthDependentDiffuser,
    ExponentialWeatherer,
)
from terrainbento.base_class import TwoLithologyErosionModel


class BasicRtSa(TwoLithologyErosionModel):
    """**BasicRtSa** model program.

    **BasicRtSa** combines the **BasicRt** and **BasicSa** programs by allowing
    for two lithologies, an "upper" layer and a "lower" layer and explicitly
    resolving a soil layer. This soil layer is produced by weathering that
    decays exponentially with soil thickness and hillslope transport is
    soil-depth dependent. Given a spatially varying contact zone elevation,
    :math:`\eta_C(x,y))`, a spatially varying soil thickness :math:`H` and a
    spatially varying bedrock elevation :math:`\eta_b`, model **BasicRtSa**
    evolves a topographic surface described by :math:`\eta` with the following
    governing equations:


    .. math::

        \eta = \eta_b + H

        \\frac{\partial H}{\partial t} = P_0 \exp (-H/H_s) - \delta (H) K A^{m} S^{n} - \\nabla q_h

        \\frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s) - (1 - \delta (H) ) K A^{m} S^{n}

        q_h = -D \left[1-\exp \left( -\\frac{H}{H_0} \\right) \\right] \\nabla \eta

        K(\eta, \eta_C ) = w K_1 + (1 - w) K_2

        w = \\frac{1}{1+\exp \left( -\\frac{(\eta -\eta_C )}{W_c}\\right)}


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`W_c` is the contact-zone width, :math:`K_1` and :math:`K_2` are the
    erodabilities of the upper and lower lithologies, and :math:`D` is the
    regolith transport parameter. :math:`w` is a weight used to calculate the
    effective erodability :math:`K(\eta, \eta_C)` based on the depth to the
    contact zone and the width of the contact zone.

    The function :math:`\delta (H)` is used to indicate that water erosion will
    act on soil where it exists, and on the underlying lithology where soil is
    absent. To achieve this, :math:`\delta (H)` is defined to equal 1 when
    :math:`H > 0` (meaning soil is present), and 0 if :math:`H = 0` (meaning the
    underlying parent material is exposed).

    Refer to the terrainbento manuscript Table XX (URL here) for parameter
    symbols, names, and dimensions.

    The weight :math:`w` promotes smoothness in the solution of erodability at a
    given point. When the surface elevation is at the contact elevation, the
    erodability is the average of :math:`K_1` and :math:`K_2`; above and below
    the contact, the erodability approaches the value of :math:`K_1` and :math:`K_2`
    at a rate related to the contact zone width. Thus, to make a very sharp
    transition, use a small value for the contact zone width.

    The **BasicRtSa** program inherits from the terrainbento
    **TwoLithologyErosionModel** base class. In addition to the parameters
    required by the base class, models built with this program require the
    following parameters.

    +------------------+-----------------------------------+
    | Parameter Symbol | Input File Parameter Name         |
    +==================+===================================+
    |:math:`m`         | ``m_sp``                          |
    +------------------+-----------------------------------+
    |:math:`n`         | ``n_sp``                          |
    +------------------+-----------------------------------+
    |:math:`K_{1}`     | ``water_erodability~upper``       |
    +------------------+-----------------------------------+
    |:math:`K_{2}`     | ``water_erodability~lower``       |
    +------------------+-----------------------------------+
    |:math:`W_{c}`     | ``contact_zone__width``           |
    +------------------+-----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter``  |
    +------------------+-----------------------------------+
    |:math:`H_{init}`  | ``soil__initial_thickness``       |
    +------------------+-----------------------------------+
    |:math:`P_{0}`     | ``soil_production__maximum_rate`` |
    +------------------+-----------------------------------+
    |:math:`H_{s}`     | ``soil_production__decay_depth``  |
    +------------------+-----------------------------------+
    |:math:`H_{0}`     | ``soil_transport__decay_depth``   |
    +------------------+-----------------------------------+

    Refer to the terrainbento manuscript Table XX (URL here) for full list of
    parameter symbols, names, and dimensions.

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
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        BasicRtSa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicRtSa**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicRtSa

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability~lower': 0.001,
        ...           'water_erodability~upper': 0.01,
        ...           'contact_zone__width': 1.0,
        ...           'lithology_contact_elevation__file_name': 'tests/data/example_contact_elevation.txt',
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'soil__initial_thickness': 2,
        ...           'soil_transport_decay_depth': 1.5,
        ...           "soil_production__maximum_rate": 0.001,
        ...           "soil_production__decay_depth": 0.7}

        Construct the model.

        >>> model = BasicRtSa(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicRtSa, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )
        self.K_rock_sp = self.get_parameter_from_exponent("water_erodability~lower") * (
            self._length_factor ** (1. - (2. * self.m))
        )
        self.K_till_sp = self.get_parameter_from_exponent("water_erodability~upper") * (
            self._length_factor ** (1. - (2. * self.m))
        )

        # Set the erodability values, these need to be double stated because a PrecipChanger may adjust them
        self.rock_erody = self.K_rock_sp
        self.till_erody = self.K_till_sp

        # Set up rock-till boundary and associated grid fields.
        self._setup_rock_and_till()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid, K_sp=self.erody, m_sp=self.m, n_sp=self.n
        )

        # Create soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.add_zeros("node", "soil__depth")

        # Create bedrock elevation field
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")

        # Set soil thickness and bedrock elevation
        initial_soil_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length

        soil_transport_decay_depth = (self._length_factor) * self.params[
            "soil_transport_decay_depth"
        ]  # has units length
        max_soil_production_rate = (self._length_factor) * self.params[
            "soil_production__maximum_rate"
        ]  # has units length per time
        soil_production_decay_depth = (self._length_factor) * self.params[
            "soil_production__decay_depth"
        ]  # has units length

        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=self.regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            max_soil_production_rate=max_soil_production_rate,
            soil_production_decay_depth=soil_production_decay_depth,
        )

    def _setup_rock_and_till(self):
        """Set up fields to handle for two layers with different erodability."""
        # Get a reference to the rock-till field\
        self._setup_contact_elevation()

        # Create field for erodability
        self.erody = self.grid.add_zeros("node", "substrate__erodability")

        # Create array for erodability weighting function
        self.erody_wt = np.zeros(self.grid.number_of_nodes)

    def _update_erodability_field(self):
        """Update erodability at each node.

        The erodability at each node is a smooth function between the rock and
        till erodabilities and is based on the contact zone width and the
        elevation of the surface relative to contact elevation.
        """

        # Update the erodability weighting function (this is "F")
        self.erody_wt[self.data_nodes] = 1.0 / (
            1.0
            + np.exp(
                -(self.z[self.data_nodes] - self.rock_till_contact[self.data_nodes])
                / self.contact_width
            )
        )

        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            erode_factor = self.boundary_handler[
                "PrecipChanger"
            ].get_erodability_adjustment_factor()
            self.till_erody = self.K_till_sp * erode_factor
            self.rock_erody = self.K_rock_sp * erode_factor

        # Calculate the effective erodibilities using weighted averaging
        self.erody[:] = (
            self.erody_wt * self.till_erody + (1.0 - self.erody_wt) * self.rock_erody
        )

    def run_one_step(self, dt):
        """Advance model **BasicRtSa** for one time-step of duration dt.

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

        6. Calculates topographic change by linear diffusion.

        7. Finalizes the step using the **ErosionModel** base class function
           **finalize__run_one_step**. This function updates all BoundaryHandlers
           by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """
        # Direct and accumulate flow
        self.flow_accumulator.run_one_step()

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
        self.eroder.run_one_step(dt, flooded_nodes=flooded, K_if_used=self.erody)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    sart = BasicRtSa(input_file=infile)
    sart.run()


if __name__ == "__main__":
    main()
