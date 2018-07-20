# coding: utf8
#! /usr/env/python
"""terrainbento **BasicDd** model program.

Erosion model program using linear diffusion, stream power with a smoothed
threshold that varies with incision depth, and discharge proportional to
drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `StreamPowerSmoothThresholdEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import StreamPowerSmoothThresholdEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicDd(ErosionModel):
    """**BasicDd** model program.

    **BasicDd** is a model program that evolves a topographic surface described
    by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -\left(KA^{m}S^{n} - \omega_{ct}\left(1-e^{-KA^{m}S^{n}/\omega_{ct}}\\right)\\right) + D\\nabla^2 \eta

    where :math:`A` is the local drainage area and :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`D` is the regolith transport
    efficiency, and :math:`\omega_{ct}` is the critical stream power needed for
    erosion to occur. :math:`\omega_{ct}` changes through time as it increases
    with cumulative incision depth:

    .. math::

        \omega_{ct}\left(x,y,t\\right) = \mathrm{max}\left(\omega_c +  b D_I\left(x, y, t\\right), \omega_c \\right)

    where :math:`\omega_c` is the threshold when no incision has taken place,
    :math:`b` is the rate at which the threshold increases with incision depth,
    and :math:`D_I` is the cumulative incision depth at location
    :math:`\left(x,y\\right)` and time :math:`t`.

    The **BasicDd** program inherits from the terrainbento **ErosionModel** base
    class. In addition to the parameters required by the base class, models
    built with this program require the following parameters.

    +--------------------+-------------------------------------------------+
    | Parameter Symbol   | Input File Parameter Name                       |
    +====================+=================================================+
    |:math:`m`           | ``m_sp``                                        |
    +--------------------+-------------------------------------------------+
    |:math:`n`           | ``n_sp``                                        |
    +--------------------+-------------------------------------------------+
    |:math:`K`           | ``water_erodability``                           |
    +--------------------+-------------------------------------------------+
    |:math:`\omega_{c}`  | ``water_erosion_rule__threshold``               |
    +--------------------+-------------------------------------------------+
    |:math:`b`           | ``water_erosion_rule__thresh_depth_derivative`` |
    +--------------------+-------------------------------------------------+
    |:math:`D`           | ``regolith_transport_parameter``                |
    +--------------------+-------------------------------------------------+

    Refer to the terrainbento manuscript Table XX (URL here) for full list of
    parameter symbols, names, and dimensions.
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
        BasicDd : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicDd**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicDd

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           "water_erosion_rule__threshold": 0.01,
        ...           'water_erosion_rule__thresh_depth_derivative': 0.01}

        Construct the model.

        >>> model = BasicDd(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicDd, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        if float(self.params["n_sp"]) != 1.0:
            raise ValueError("Model BasicDd only supports n equals 1.")

        # Get Parameters and convert units if necessary:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self.get_parameter_from_exponent("water_erodability") * (
            self._length_factor ** (1. - (2. * self.m))
        )

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

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros("node", "water_erosion_rule__threshold")
        self.threshold[:] = self.threshold_value

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerSmoothThresholdEroder(
            self.grid,
            m_sp=self.m,
            n_sp=self.n,
            K_sp=self.K,
            threshold_sp=self.threshold,
        )

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params[
            "water_erosion_rule__thresh_depth_derivative"
        ]

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def update_erosion_threshold_values(self):
        """Update the erosion threshold at each node based on cumulative
        incision so far using:

        .. math::

            \omega_{ct}\left(x,y,t\\right) = \mathrm{max}\left(\omega_c + \\
            b D_I\left(x, y, t\\right), \omega_c \\right)

        where :math:`\omega_c` is the threshold when no incision has taken place,
        :math:`b` is the rate at which the threshold increases with incision depth,
        and :math:`D_I` is the cumulative incision depth at location
        :math:`\left(x,y\\right)` and time :math:`t`.

        """

        # Set the erosion threshold.
        #
        # Note that a minus sign is used because cum ero depth is negative for
        # erosion, positive for deposition.
        # The second line handles the case where there is growth, in which case
        # we want the threshold to stay at its initial value rather than
        # getting smaller.
        cum_ero = self.grid.at_node["cumulative_erosion__depth"]
        cum_ero[:] = self.z - self.grid.at_node["initial_topographic__elevation"]
        self.threshold[:] = self.threshold_value - (
            self.thresh_change_per_depth * cum_ero
        )
        self.threshold[self.threshold < self.threshold_value] = self.threshold_value

    def run_one_step(self, dt):
        """Advance model **BasicDd** for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited, threshold-modified erosion by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the **ErosionModel** base class function
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

        # Calculate the new threshold values given cumulative erosion
        self.update_erosion_threshold_values()

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

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

    ldsp = BasicDd(input_file=infile)
    ldsp.run()


if __name__ == "__main__":
    main()
