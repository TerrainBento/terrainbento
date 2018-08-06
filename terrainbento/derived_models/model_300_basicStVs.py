# coding: utf8
#! /usr/env/python
"""
terrainbento Model **BasicStVs** program.

Erosion model program using linear diffusion and stream power. Precipitation is
modeled as a stochastic process. Discharge is calculated from precipitation
using a simple variable source-area formulation.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_

Landlab components used: FlowRouter, DepressionFinderAndRouter,
PrecipitationDistribution, StreamPowerEroder, LinearDiffuser

"""

import numpy as np

from landlab.components import StreamPowerEroder, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel


class BasicStVs(StochasticErosionModel):
    """
    **BasicStVs** model program.

    **BasicStVs** is a model program that uses a stochastic treatment of runoff
    and discharge, using a variable source area runoff generation model.
    THe model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{q}\hat{Q}^{m}S^{n} + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable) and :math:`S` is the local
    slope gradient. Refer to the terrainbento manuscript Table XX (URL here)
    for parameter symbols, names, and dimensions.

    This model iterates through a sequence of storm and interstorm periods.
    Given a storm precipitation intensity $P$, the discharge $Q$ [L$^3$/T]
    is calculated using:

    .. math::

        Q = PA - T\lambda S [1 - \exp (-PA/T\lambda S) ]

    where :math:`T = K_sH` is the soil transmissivity, :math:`H` is soil
    thickness, :math:`K_s` is hydraulic conductivity, and :math:`\lambda` is
    cell width.

    **BasicStVs** inherits from the terrainbento **StochasticErosionModel**
    base class. In addition to the parameters required by the base class,
    models built with this program require the following parameters:

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K_q`       | ``water_erodability~stochastic`` |
    +------------------+----------------------------------+
    |:math:`H`         | ``soil__initial_thickness``      |
    +------------------+----------------------------------+
    |:math:`H`         | ``soil__initial_thickness``      |
    +------------------+----------------------------------+
    |:math:`K_s`       | ``hydraulic_conductivity``       |
    +------------------+----------------------------------+

    For information about the stochastic precipitation and runoff model used,
    see the documentation for **BasicSt** and the base class
    **StochasticErosionModel**.

    Note that there is no unique single runoff rate in this model, because
    runoff rate varies in space. Therefore, the class variable
    runoff_rate (which contains a single value per event) should be ignored.
    """

    def __init__(self, input_file=None, params=None, OutputWriters=None):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        BasicStVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicStVs**. Note that a YAML input file can be used instead
        of a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicStVs

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability~stochastic': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'opt_stochastic_duration': False,
        ...           'number_of_sub_time_steps': 1,
        ...           'rainfall_intermittency_factor': 0.5,
        ...           'rainfall__mean_rate': 1.0,
        ...           'rainfall__shape_factor': 1.0,
        ...           'infiltration_capacity': 1.0,
        ...           'random_seed': 0,
        ...           'soil__initial_thickness': 2.0,
        ...           'hydraulic_conductivity': 0.1}

        Construct the model.

        >>> model = BasicStVs(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicStVs, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )
        # Get Parameters:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self.get_parameter_from_exponent("water_erodability~stochastic") * (
            self._length_factor ** ((3. * self.m) - 1)
        )  # K stochastic has units of [=] T^{m-1}/L^{3m-1}

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        soil_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length
        K_hydraulic_conductivity = (self._length_factor) * self.params[
            "hydraulic_conductivity"
        ]  # has units length per time

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Add a field for subsurface discharge
        self.qss = self.grid.add_zeros("node", "subsurface_water__discharge")

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.trans = K_hydraulic_conductivity * soil_thickness

        if self.trans <= 0.0:
            raise ValueError("BasicStVs: Transmissivity must be > 0")

        self.tlam = self.trans * self.grid._dx  # assumes raster

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Keep a reference to drainage area and steepest-descent slope
        self.area = self.grid.at_node["drainage_area"]
        self.slope = self.grid.at_node["topographic__steepest_slope"]

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
            self.grid, use_Q=self.discharge, K_sp=self.K, m_sp=self.m, n_sp=self.m
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def calc_runoff_and_discharge(self):
        """Calculate runoff rate and discharge; return runoff."""

        # Here's the total (surface + subsurface) discharge
        pa = self.rain_rate * self.area

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = self.tlam * self.slope[np.where(self.slope > 0.0)[0]]

        # Subsurface discharge: zero where slope is flat
        self.qss[np.where(self.slope <= 0.0)[0]] = 0.0
        self.qss[np.where(self.slope > 0.0)[0]] = tls * (
            1.0 - np.exp(-pa[np.where(self.slope > 0.0)[0]] / tls)
        )

        # Surface discharge = total minus subsurface
        #
        # Note that roundoff errors can sometimes produce a tiny negative
        # value when qss and pa are close; make sure these are set to 0
        self.discharge[:] = pa - self.qss
        self.discharge[self.discharge < 0.0] = 0.0

        return np.nan

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
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

        # Handle water erosion
        self.handle_water_erosion(dt, flooded)

        # Do some soil creep
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

    dm = BasicStVs(input_file=infile)
    dm.run()


if __name__ == "__main__":
    main()
