# coding: utf8
#! /usr/env/python
"""
terrainbento **BasicDdSt** model program.

Erosion model program using linear diffusion, smoothly thresholded stream
power, and stochastic discharge with a smoothed infiltration capacity
threshold. The program differs from BasicStTh in that the threshold value
depends on cumulative incision depth, and so can vary in space and time.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerSmoothThresholdEroder`
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, StreamPowerSmoothThresholdEroder
from terrainbento.base_class import StochasticErosionModel


class BasicDdSt(StochasticErosionModel):
    """
    **BasicDdSt** model program.

    **BasicDdSt** is a model program that uses a stochastic treatment of runoff
    and discharge, and includes an erosion threshold in the water erosion law.
    The threshold depends on cumulative incision depth, and therefore can vary
    in space and time.

    THe model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -[K_{q}\hat{Q}^{m}S^{n} - \omega_{ct}\left(1-e^{-K_{q}\hat{Q}^{m}S^{n}/\omega_{ct}}\\right)\\right)] + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable) and :math:`S` is the local
    slope gradient. :math:`m` and :math:`n` are the discharge and slope
    exponent, respectively, :math:`\omega_c` is the critical stream power
    required for erosion to occur, and :math:`D` is the regolith transport
    parameter.
    
        :math:`\omega_{ct}` may change through time as it increases with cumulative
    incision depth:

    .. math::

        \omega_{ct}\left(x,y,t\\right) = \mathrm{max}\left(\omega_c + b D_I\left(x, y, t\\right), \omega_c \\right)

    where :math:`\omega_c` is the threshold when no incision has taken place,
    :math:`b` is the rate at which the threshold increases with incision depth,
    and :math:`D_I` is the cumulative incision depth at location
    :math:`\left(x,y\\right)` and time :math:`t`.
    
    Refer to the terrainbento manuscript Table XX (URL here)
    for parameter symbols, names, and dimensions.

    **BasicDdSt** inherits from the terrainbento **StochasticErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters:

    +--------------------+----------------------------------+
    | Parameter Symbol   | Input File Parameter Name        |
    +====================+==================================+
    |:math:`m`           | ``m_sp``                         |
    +--------------------+----------------------------------+
    |:math:`n`           | ``n_sp``                         |
    +--------------------+----------------------------------+
    |:math:`K_q`         | ``water_erodability~stochastic`` |
    +--------------------+----------------------------------+
    |:math:`\omega_{c0}` | ``water_erosion_rule__threshold``|
    +--------------------+----------------------------------+
    |:math:`D`           | ``regolith_transport_parameter`` |
    +--------------------+----------------------------------+
    |:math:`I_m`         | ``infiltration_capacity``        |
    +--------------------+----------------------------------+

    For information about the stochastic precipitation and runoff model used,
    see the documentation for **BasicSt** and the base class
    **StochasticErosionModel**.
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
        BasicDdSt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDdSt**. Note that a YAML input file can be used instead
        of a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicDdSt

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
        ...           'water_erosion_rule__threshold': 1.0,
        ...           'thresh_change_per_depth': 0.1,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'opt_stochastic_duration': False,
        ...           'number_of_sub_time_steps': 1,
        ...           'rainfall_intermittency_factor': 0.5,
        ...           'rainfall__mean_rate': 1.0,
        ...           'rainfall__shape_factor': 1.0,
        ...           'infiltration_capacity': 1.0,
        ...           'random_seed': 0}

        Construct the model.

        >>> model = BasicDdSt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicDdSt, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )

        # Get Parameters:
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

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        self.threshold_value = self._length_factor * self.get_parameter_from_exponent(
            "water_erosion_rule__threshold"
        )  # has units length/time

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params["thresh_change_per_depth"]

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Get the infiltration-capacity parameter
        # has units length per time
        self.infilt = (self._length_factor) * self.params["infiltration_capacity"]

        # Keep a reference to drainage area
        self.area = self.grid.at_node["drainage_area"]

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros("node", "water_erosion_rule__threshold")
        self.threshold[:] = self.threshold_value

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params["thresh_change_per_depth"]

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            m_sp=self.m,
            n_sp=self.n,
            K_sp=self.K,
            use_Q=self.discharge,
            threshold_sp=self.threshold,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def update_threshold_field(self):
        """Update the threshold based on cumulative erosion depth."""
        cum_ero = self.grid.at_node["cumulative_elevation_change"]
        cum_ero[:] = self.z - self.grid.at_node["initial_topographic__elevation"]
        self.threshold[:] = self.threshold_value - (
            self.thresh_change_per_depth * cum_ero
        )
        self.threshold[self.threshold < self.threshold_value] = self.threshold_value

    def _pre_water_erosion_steps(self):
        self.update_threshold_field()

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

    em = BasicDdSt(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
