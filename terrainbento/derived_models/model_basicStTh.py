# coding: utf8
# !/usr/env/python
"""terrainbento **BasicStTh** model program.

Erosion model program using linear diffusion, smoothly thresholded stream
power, and stochastic discharge with a smoothed infiltration capacity
threshold.

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

_REQUIRED_FIELDS = ["topographic__elevation"]


class BasicStTh(StochasticErosionModel):
    """
    **BasicStTh** model program.

    **BasicStTh** is a model program that uses a stochastic treatment of runoff
    and discharge, and includes an erosion threshold in the water erosion law.
    The model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -(K_{q}\hat{Q}^{m}S^{n} - \omega_c) + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable) and :math:`S` is the local
    slope gradient. :math:`m` and :math:`n` are the discharge and slope
    exponent, respectively, :math:`\omega_c` is the critical stream power
    required for erosion to occur, and :math:`D` is the regolith transport
    parameter.

    **BasicSt** inherits from the terrainbento **StochasticErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +------------------+----------------------------------+
    | Parameter Symbol | Input File Parameter Name        |
    +==================+==================================+
    |:math:`m`         | ``m_sp``                         |
    +------------------+----------------------------------+
    |:math:`n`         | ``n_sp``                         |
    +------------------+----------------------------------+
    |:math:`K_q`       | ``water_erodability~stochastic`` |
    +------------------+----------------------------------+
    |:math:`\omega_c`  | ``water_erosion_rule__threshold``|
    +------------------+----------------------------------+
    |:math:`D`         | ``regolith_transport_parameter`` |
    +------------------+----------------------------------+
    |:math:`I_m`       | ``infiltration_capacity``        |
    +------------------+----------------------------------+

    Refer to the terrainbento manuscript Table 5 (URL to manuscript when
    published) for full list of parameter symbols, names, and dimensions.

    For information about the stochastic precipitation and runoff model used,
    see the documentation for **BasicSt** and the base class
    **StochasticErosionModel**.
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
        BasicStTh : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicStTh**. For more detailed examples, including steady-state
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
        ...           "water_erodability~stochastic": 0.001,
        ...           "water_erosion_rule__threshold": 0.2,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0,
        ...           "opt_stochastic_duration": False,
        ...           "number_of_sub_time_steps": 1,
        ...           "rainfall_intermittency_factor": 0.5,
        ...           "rainfall__mean_rate": 1.0,
        ...           "rainfall__shape_factor": 1.0,
        ...           "infiltration_capacity": 1.0,
        ...           "random_seed": 0}

        Construct the model.

        >>> model = BasicStTh(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicStTh, self).__init__(clock, grid, **kwargs)

        # Get Parameters:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self._get_parameter_from_exponent(
            "water_erodability~stochastic"
        ) * (
            self._length_factor ** ((3. * self.m) - 1)
        )  # K stochastic has units of [=] T^{m-1}/L^{3m-1}

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self._get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        #  threshold has units of  Length per Time which is what
        # StreamPowerSmoothThresholdEroder expects
        threshold = self._length_factor * self._get_parameter_from_exponent(
            "water_erosion_rule__threshold"
        )  # has units length/time

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Get the infiltration-capacity parameter
        # has units length per time
        self.infilt = (self._length_factor) * self.params[
            "infiltration_capacity"
        ]

        # Keep a reference to drainage area
        self.area = self.grid.at_node["drainage_area"]

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            K_sp=self.K,
            m_sp=self.m,
            n_sp=self.n,
            threshold_sp=threshold,
            use_Q="surface_water__discharge",
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, step):
        """
        Advance model for one time-step of duration step.
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

    em = BasicStTh(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
