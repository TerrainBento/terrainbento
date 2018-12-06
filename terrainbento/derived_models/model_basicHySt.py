# coding: utf8
# !/usr/env/python
"""
terrainbento Model **BasicHySt** program.

Erosion model program using linear diffusion for gravitational mass transport,
and an entrainment-deposition law for water erosion and deposition. Discharge
is calculated from drainage area, infiltration capacity (a parameter), and
precipitation rate, which is a stochastic variable.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
    5. `PrecipitationDistribution <http://landlab.readthedocs.io/en/latest/landlab.components.html#landlab.components.PrecipitationDistribution>`_
"""

import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import StochasticErosionModel


class BasicHySt(StochasticErosionModel):
    """
    **BasicHySt** model program.

    **BasicHySt** is a model program that uses a stochastic treatment of runoff
    and discharge, and includes an erosion threshold in the water erosion law.
    THe model evolves a topographic surface, :math:`\eta (x,y,t)`,
    with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -E(\hat{Q}) + D_s(\hat{Q}) + D\\nabla^2 \eta

    where :math:`\hat{Q}` is the local stream discharge (the hat symbol
    indicates that it is a random-in-time variable), :math:`E` is the bed erosion
    (entrainment) rate due to fluid entrainment, :math:`D_s` is the deposition
    rate of sediment settling out of active transport, and :math:`D` is the
    regolith transport parameter.

    **BasicHySt** inherits from the terrainbento **StochasticErosionModel**
    base class. In addition to the parameters required by the base class, models
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
    |:math:`V_s`       | ``v_s``                          |
    +------------------+----------------------------------+
    |:math:`F_f`       | ``fraction_fines``               |
    +------------------+----------------------------------+
    |:math:`\phi`      | ``sediment_porosity``            |
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
        BasicHySt : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHySt**. Note that a YAML input file can be used instead
        of a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicHySt

        Set up a parameters variable.

        >>> params = {"model_grid": "RasterModelGrid",
        ...           "dt": 1,
        ...           "output_interval": 2.,
        ...           "run_duration": 200.,
        ...           "number_of_node_rows" : 6,
        ...           "number_of_node_columns" : 9,
        ...           "node_spacing" : 10.0,
        ...           "regolith_transport_parameter": 0.001,
        ...           "water_erodability~stochastic": 0.001,
        ...           "m_sp": 0.5,
        ...           "n_sp": 1.0,
        ...           "opt_stochastic_duration": False,
        ...           "number_of_sub_time_steps": 1,
        ...           "rainfall_intermittency_factor": 0.5,
        ...           "rainfall__mean_rate": 1.0,
        ...           "rainfall__shape_factor": 1.0,
        ...           "infiltration_capacity": 1.0,
        ...           "random_seed": 0,
        ...           "v_s": 0.01,
        ...           "fraction_fines": 0.1,
        ...           "sediment_porosity": 0.3,
        ...           "solver": "adaptive"}

        Construct the model.

        >>> model = BasicHySt(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel"s init
        super(BasicHySt, self).__init__(
            input_file=input_file, params=params, OutputWriters=OutputWriters
        )

        # Get Parameters:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self._get_parameter_from_exponent(
            "water_erodability~stochastic"
        ) * (
            self._length_factor ** ((3. * self.m) - 1)
        )  # K stochastic has units of [=] T^{m-1}/L^{3m-1}

        regolith_transport_parameter = (
            self._length_factor ** 2
        ) * self._get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # L^2/T

        v_s = (self._length_factor) * self._get_parameter_from_exponent(
            "v_s"
        )  # has units length per time

        # instantiate rain generator
        self.instantiate_rain_generator()

        # Add a field for discharge
        self.discharge = self.grid.at_node["surface_water__discharge"]

        # Get the infiltration-capacity parameter
        # has units length per time
        self.infilt = (self._length_factor) * self.params[
            "infiltration_capacity"
        ]

        # Run flow routing and lake filler
        self.flow_accumulator.run_one_step()

        # Keep a reference to drainage area
        self.area = self.grid.at_node["drainage_area"]

        # Handle solver option
        solver = self.params.get("solver", "basic")

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            F_f=self.params["fraction_fines"],
            phi=self.params["sediment_porosity"],
            v_s=v_s,
            m_sp=self.m,
            n_sp=self.n,
            discharge_field="surface_water__discharge",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

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

    em = BasicHySt(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
