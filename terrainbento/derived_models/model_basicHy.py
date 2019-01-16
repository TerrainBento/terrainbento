# coding: utf8
# !/usr/env/python
"""terrainbento **BasicHy** model program.

Erosion model program using linear diffusion, stream-power-driven sediment
erosion and mass conservation, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicHy(ErosionModel):
    r"""**BasicHy** model program.

    **BasicHy** is a model program that evolves a topographic surface described
    by :math:`\eta` with the following governing equation:


    .. math::

        \\\frac{\partial \eta}{\partial t} = -KA^{m}S^{n} + \\\frac{V Q_s}{rA \left(1-\phi\\right)} + D\nabla^2 \eta


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`V` is effective sediment
    settling velocity, :math:`Q_s` is volumetric sediment flux, :math:`r` is
    a runoff rate which presently can only be 1.0, :math:`\phi` is sediment
    porosity, :math:`D` is the regolith transport efficiency, :math:`H` is soil
    depth, and :math:`H_*` is the bedrock roughness length scale.

    If you have use cases for which :math:`r=1.0` is not sufficient, please make
    an Issue on GitHub.

    The **BasicHy** program inherits from the terrainbento **ErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K`         | ``water_erodability``            |
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`V`         | ``settling_velocity``            |
    +------------------+----------------------------------+
    |:math:`F_f`       | ``fraction_fines``               |
    +------------------+----------------------------------+
    |:math:`\phi`      | ``sediment_porosity``            |
    +------------------+----------------------------------+

    Refer to
    `Barnhart et al. (2019) <https://www.geosci-model-dev-discuss.net/gmd-2018-204/>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodability=0.0001,
        regolith_transport_parameter=0.1,
        settling_velocity=0.001,
        sediment_porosity=0.3,
        fraction_fines=0.5,
        solver="basic",
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
            :py:class:`~terrainbento.base_class.erosion_model.ErosionModel`.

        Returns
        -------
        BasicHy : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHy**. For more detailed examples, including steady-state
        test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicHy
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")

        Construct the model.

        >>> model = BasicHy(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel"s init
        super(BasicHy, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Get Parameters
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodability

        # Instantiate a Space component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            phi=sediment_porosity,
            F_f=fraction_fines,
            v_s=settling_velocity,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, step):
        """Advance model **BasicHy** for one time-step of duration step.

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
            self.eroder.K = (
                self.K
                * self.boundary_handlers[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(
            step,
            flooded_nodes=flooded,
            dynamic_dt=True,
            flow_director=self.flow_accumulator.flow_director,
        )

        # Do some soil creep
        self.diffuser.run_one_step(step)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(step)


def main():  # pragma: no cover
    """Execute model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    ha = BasicHy(input_file=infile)
    ha.run()


if __name__ == "__main__":
    main()
