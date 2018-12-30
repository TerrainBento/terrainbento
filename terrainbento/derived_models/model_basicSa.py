# coding: utf8
# !/usr/env/python
"""terrainbento **BasicSa** model program.

Erosion model using depth-dependent linear diffusion with a soil layer, basic
stream power, and discharge proportional to drainage area.


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
    FastscapeEroder,
)
from terrainbento.base_class import ErosionModel

_REQUIRED_FIELDS = ["topographic__elevation", "soil__depth"]


class BasicSa(ErosionModel):
    r"""**BasicSa** model program.

    **BasicSa** is a model program that explicitly resolves a soil layer. This
    soil layer is produced by weathering that decays exponentially with soil
    thickness and hillslope transport is soil-depth dependent. Given a spatially
    varying soil thickness :math:`H` and a spatially varying bedrock elevation
    :math:`\eta_b`, model **BasicSa** evolves a topographic surface described by
    :math:`\eta` with the following governing equations:


    .. math::

        \eta = \eta_b + H

        \\frac{\partial H}{\partial t} = P_0 \exp (-H/H_s) - \delta (H) K A^{M} S^{N} -\\nabla q_h

        \\frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s) - (1 - \delta (H) ) K A^{m} S^{N}

        q_h = -D \left[1-\exp \left( -\\frac{H}{H_0} \\right) \\right] \\nabla \eta


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`D` is the regolith transport
    parameter, :math:`H` is the regolith thickness, :math:`H_s`
    is the sediment production decay depth, :math:`P_0` is the maximum sediment
    production rate, and :math:`H_0` is the sediment transport decay depth. :math:`q_s`
    represents the hillslope sediment flux per unit width.

    The function :math:`\delta (H)` is used to indicate that water erosion will
    act on soil where it exists, and on the underlying lithology where soil is
    absent. To achieve this, :math:`\delta (H)` is defined to equal 1 when
    :math:`H > 0` (meaning soil is present), and 0 if :math:`H = 0` (meaning the
    underlying parent material is exposed).

    The **BasicSa** program inherits from the terrainbento **ErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+-----------------------------------+
    | Parameter Symbol | Input File Parameter Name         |
    +==================+===================================+
    |:math:`m`         | ``m_sp``                          |
    +------------------+-----------------------------------+
    |:math:`n`         | ``n_sp``                          |
    +------------------+-----------------------------------+
    |:math:`K`         | ``water_erodability``             |
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

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.

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
        BasicSa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSa**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, Basic
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

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
        ...           "soil__initial_thickness": 0.0,
        ...           "soil_transport_decay_depth": 0.2,
        ...           "soil_production__maximum_rate": 0.001,
        ...           "soil_production__decay_depth": 0.1,
        ...           "water_erodability": 0.001,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0}

        Construct the model.

        >>> model = BasicSa(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel"s init
        super(BasicSa, self).__init__(clock, grid, **kwargs)

        # Get Parameters and convert units if necessary:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability * (
            self._length_factor ** (1. - (2. * self.m))
        )

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * regolith_transport_parameter
        initial_soil_thickness = (
            self._length_factor
        ) * soil__initial_thickness
        soil_transport_decay_depth = (
            self._length_factor
        ) * soil_transport_decay_depth
        max_soil_production_rate = (
            self._length_factor
        ) * soil_production__maximum_rate
        soil_production_decay_depth = (
            self._length_factor
        ) * soil_production__decay_depth

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
            discharge_name="surface_water__discharge",
        )

        # Create soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.add_zeros("node", "soil__depth")

        # Create bedrock elevation field
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")

        # Set soil thickness and bedrock elevation
        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=max_soil_production_rate,
            soil_production__decay_depth=soil_production_decay_depth,
        )

    def run_one_step(self, step):
        """Advance model **BasicSa** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited erosion by water.

        5. Produces soil and calculates soil depth with exponential weathering.

        6. Calculates topographic change by depth-dependent linear diffusion.

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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handlers:
            self.eroder.K = (
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
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

    ldsp = BasicSa(input_file=infile)
    ldsp.run()


if __name__ == "__main__":
    main()
