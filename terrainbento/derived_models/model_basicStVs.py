# coding: utf8
# !/usr/env/python
"""terrainbento Model **BasicStVs** program.

Erosion model program using linear diffusion and stream power. Precipitation is
modeled as a stochastic process. Discharge is calculated from precipitation
using a simple variable source-area formulation.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel


class BasicStVs(StochasticErosionModel):
    r"""**BasicStVs** model program.

    This model program uses a stochastic treatment of runoff and discharge,
    using a variable source area runoff generation model. It combines
    :py:class:`BasicSt` and :py:class:`BasicVs`. The model evolves a
    topographic surface, :math:`\eta (x,y,t)`, with the following governing
    equation:

    .. math::

        \frac{\partial \eta}{\partial t} = -K_{q}\hat{Q}^{m}S^{n}
                                           + D\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable) and :math:`S` is the local
    slope gradient. :math:`m` and :math:`n` are the discharge and slope
    exponent, respectively, :math:`K` is the erodibility by water, and
    :math:`D` is the regolith transport parameter.

    This model iterates through a sequence of storm and interstorm periods.
    Given a storm precipitation intensity :math:`P`, the discharge :math:`Q`.
    is calculated using:

    .. math::

        Q = PA - T\lambda S [1 - \exp (-PA/T\lambda S) ]

    where :math:`T = K_sH` is the soil transmissivity, :math:`H` is soil
    thickness, :math:`K_s` is hydraulic conductivity, and :math:`\lambda` is
    cell width.

    Refer to
    `Barnhart et al. (2019) <https://doi.org/10.5194/gmd-12-1267-2019>`_
    Table 5 for full list of parameter symbols, names, and dimensions.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
    """

    _required_fields = ["topographic__elevation"]

    def __init__(
        self,
        clock,
        grid,
        m_sp=0.5,
        n_sp=1.0,
        water_erodibility=0.0001,
        regolith_transport_parameter=0.1,
        hydraulic_conductivity=0.1,
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
        water_erodibility : float, optional
            Water erodibility (:math:`K`). Default is 0.0001.
        regolith_transport_parameter : float, optional
            Regolith transport efficiency (:math:`D`). Default is 0.1.
        infiltration_capacity: float, optional
            Infiltration capacity (:math:`I_m`). Default is 1.0.
        hydraulic_conductivity : float, optional
            Hydraulic conductivity (:math:`K_{sat}`). Default is 0.1.
        **kwargs :
            Keyword arguments to pass to :py:class:`StochasticErosionModel`.
            These arguments control the discharge :math:`\hat{Q}`.

        Returns
        -------
        BasicStVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicStVs**. For more detailed examples, including
        steady-state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from landlab import RasterModelGrid
        >>> from landlab.values import random
        >>> from terrainbento import Clock, BasicStVs
        >>> clock = Clock(start=0, stop=100, step=1)
        >>> grid = RasterModelGrid((5,5))
        >>> _ = random(grid, "topographic__elevation")
        >>> _ = random(grid, "soil__depth")

        Construct the model.

        >>> model = BasicStVs(clock, grid)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicStVs, self).__init__(clock, grid, **kwargs)

        # verify correct fields are present.
        self._verify_fields(self._required_fields)

        # Get Parameters:
        self.m = m_sp
        self.n = n_sp
        self.K = water_erodibility

        soil_thickness = self.grid.at_node["soil__depth"]

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for subsurface discharge
        self.qss = self.grid.add_zeros("node", "subsurface_water__discharge")

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.trans = hydraulic_conductivity * soil_thickness

        if np.any(self.trans) <= 0.0:
            raise ValueError("BasicStVs: Transmissivity must be > 0")

        self.tlam = self.trans * self.grid._dx  # assumes raster

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.m,
            discharge_name="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""

        # Here"s the total (surface + subsurface) discharge
        pa = self.rain_rate * self.grid.at_node["drainage_area"]

        # slope > 0
        active_nodes = np.where(
            self.grid.at_node["topographic__steepest_slope"] > 0.0
        )[0]

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = (
            self.tlam[active_nodes]
            * self.grid.at_node["topographic__steepest_slope"][active_nodes]
        )

        # Subsurface discharge: zero where slope is flat
        self.qss[active_nodes] = 0.0
        self.qss[active_nodes] = tls * (1.0 - np.exp(-pa[active_nodes] / tls))

        # Surface discharge = total minus subsurface
        #
        # Note that roundoff errors can sometimes produce a tiny negative
        # value when qss and pa are close; make sure these are set to 0
        self.grid.at_node["surface_water__discharge"][:] = pa - self.qss
        self.grid.at_node["surface_water__discharge"][
            self.grid.at_node["surface_water__discharge"] < 0.0
        ] = 0.0

        return np.nan

    def run_one_step(self, step):
        """Advance model **BasicStVs** for one time-step of duration step.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a :py:mod:`PrecipChanger` is an active boundary handler
           and if so, uses it to modify the erodibility by water.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the :py:mod:`ErosionModel` base class
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

        # Handle water erosion
        self.handle_water_erosion(step, flooded)

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

    dm = BasicStVs.from_file(infile)
    dm.run()


if __name__ == "__main__":
    main()
