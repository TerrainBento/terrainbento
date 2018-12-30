# coding: utf8
# !/usr/env/python
"""terrainbento model **BasicHySa** program.

Erosion model program using exponential weathering, soil-depth-dependent
linear diffusion, stream-power-driven sediment erosion, mass conservation, and
bedrock erosion, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `Space <http://landlab.readthedocs.io/en/release/landlab.components.space.html>`_
    4. `DepthDependentDiffuser <http://http://landlab.readthedocs.io/en/release/landlab.components.depth_dependent_diffusion.html>`_
    5. `ExponentialWeatherer <http://http://landlab.readthedocs.io/en/release/landlab.components.weathering.html>`_
"""

import numpy as np

from landlab.components import (
    DepthDependentDiffuser,
    ExponentialWeatherer,
    Space,
)
from terrainbento.base_class import ErosionModel

_REQUIRED_FIELDS = ["topographic__elevation", "soil__depth"]


class BasicHySa(ErosionModel):
    """Model **BasicHySa** program.

    Model **BasicHySa** is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:


    .. math::

        \eta = \eta_b + H

        \\frac{\partial H}{\partial t} = P_0 \exp (-H/H_s) + \\frac{V_s Q_s}{Ar\left(1 - \phi \\right)} - K_s A^{m}S^{n} (1 - e^{-H/H_*}) -\\nabla q_h

        \\frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s) - K_r A^{m}S^{n} e^{-H/H_*}

        Q_s = \int_0^A \left(K_s A^{m}S^{n} (1-e^{-H/H_*}) + K_r (1-F_f) A^{m}S^{n} e^{-H/H_*} - \\frac{V_s Q_s}{Ar\left(1 - \phi \\right)}\\right) dA


    where :math:`\eta_b` is the bedrock elevation, :math:`H` is the soil depth,
    :math:`P_0` is the maximum soil production rate, :math:`H_s` is the soil
    production decay depth, :math:`V_s` is effective sediment settling velocity,
    :math:`Q_s` is volumetric fluvial sediment flux, :math:`A` is the local
    drainage area, :math:`S` is the local slope, :math:`\phi` is sediment
    porosity, :math:`F_f` is the fraction of fine sediment, :math:`K_r` and :math:`K_s`
    are rock and sediment erodibility respectively, :math:`m` and :math:`n` are
    the drainage area and slope exponent parameters, :math:`H_*` is the bedrock roughness
    length scale, and :math:`r` is a runoff rate which presently can only be 1.0.
    Hillslope sediment flux per unit width :math:`q_h` is given by:


    .. math::

        q_h = -D \left[1-\exp \left( -\\frac{H}{H_0} \\right) \\right] \\nabla \eta.


    where :math:`D` is soil diffusivity and :math:`H_0` is the soil transport
    depth scale.

    The **BasicHySa** program inherits from the terrainbento **ErosionModel**
    base class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+-----------------------------------+
    | Parameter Symbol | Input File Parameter Name         |
    +==================+===================================+
    |:math:`m`         | ``m_sp``                          |
    +------------------+-----------------------------------+
    |:math:`n`         | ``n_sp``                          |
    +------------------+-----------------------------------+
    |:math:`K_r`       | ``water_erodability_rock``        |
    +------------------+-----------------------------------+
    |:math:`K_s`       | ``water_erodability_sediment``    |
    +------------------+-----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter``  |
    +------------------+-----------------------------------+
    |:math:`V_c`       | ``normalized_settling_velocity``  |
    +------------------+-----------------------------------+
    |:math:`F_f`       | ``fraction_fines``                |
    +------------------+-----------------------------------+
    |:math:`\phi`      | ``sediment_porosity``             |
    +------------------+-----------------------------------+
    |:math:`H_{init}`  | ``soil__initial_thickness``       |
    +------------------+-----------------------------------+
    |:math:`P_{0}`     | ``soil_production__maximum_rate`` |
    +------------------+-----------------------------------+
    |:math:`H_{s}`     | ``soil_production__decay_depth``  |
    +------------------+-----------------------------------+
    |:math:`H_{0}`     | ``soil_transport__decay_depth``   |
    +------------------+-----------------------------------+
    |:math:`H_{*}`     | ``roughness__length_scale``       |
    +------------------+-----------------------------------+

    A value for the parameter ``solver`` can also be used to indicate if the
    default internal timestepping is used for the **Space** component or if an
    adaptive internal timestep is used. Refer to the **Space** documentation for
    details.

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
        solver="basic",
        **kwargs
    ):
        """
        Parameters
        ----------


        Returns
        -------
        BasicHySa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHySa**. For more detailed examples, including steady-state
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
        1.0

        >>> params = {"model_grid": "RasterModelGrid",
        ...           "clock": {"step": 1,
        ...                     "output_interval": 2.,
        ...                     "stop": 200.},
        ...           "number_of_node_rows" : 6,
        ...           "number_of_node_columns" : 9,
        ...           "node_spacing" : 10.0,
        ...           "regolith_transport_parameter": 0.001,
        ...           "water_erodability_rock": 0.001,
        ...           "water_erodability_sediment": 0.001,
        ...           "sp_crit_br": 0,
        ...           "sp_crit_sed": 0,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0,
        ...           "v_sc": 0.01,
        ...           "sediment_porosity": 0,
        ...           "fraction_fines": 0,
        ...           "roughness__length_scale": 0.1,
        ...           "solver": "basic",
        ...           "soil_transport_decay_depth": 1,
        ...           "soil_production__maximum_rate": 0.0001,
        ...           "soil_production__decay_depth": 0.5,
        ...           "soil__initial_thickness": 1.0}

        Construct the model.

        >>> model = BasicHySa(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicHySa, self).__init__(clock, grid, **kwargs)

        self.m = m_sp
        self.n = n_sp
        self.K_br = (water_erodability_rock) * (
            self._length_factor ** (1. - (2. * self.m))
        )
        self.K_sed = (water_erodability_sediment) * (
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
        ) * soil_production__decay_dept

        # Instantiate a SPACE component
        self.eroder = Space(
            self.grid,
            K_sed=self.K_sed,
            K_br=self.K_br,
            sp_crit_br=sp_crit_br,
            sp_crit_sed=sp_crit_sed,
            F_f=fraction_fines,
            phi=sediment_porosity,
            H_star=roughness__length_scale,
            v_s=v_sc,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Get soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.at_node["soil__depth"]

        # Get bedrock elevation field
        bedrock_elev = self.grid.at_node["bedrock__elevation"]

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

        self.grid.at_node["soil__depth"][:] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["bedrock__elevation"]
        )

    def run_one_step(self, step):
        """Advance model **BasicHySa** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
        not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
        so, uses it to modify the erodability by water.

        4. Calculates erosion and deposition by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the **ErosionModel** base class function
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
            erode_factor = self.boundary_handlers[
                "PrecipChanger"
            ].get_erodability_adjustment_factor()
            self.eroder.K_sed = self.K_sed * erode_factor
            self.eroder.K_br = self.K_br * erode_factor

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

        # Check stability
        self.check_stability()

    def check_stability(self):
        """Check model stability and exit if unstable."""
        fields = self.grid.at_node.keys()
        for f in fields:
            if np.any(np.isnan(self.grid.at_node[f])) or np.any(
                np.isinf(self.grid.at_node[f])
            ):
                raise SystemExit(
                    "terrainbento ModelHySa: Model became unstable"
                )


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    hysa = BasicHySa(input_file=infile)
    hysa.run()


if __name__ == "__main__":
    main()
