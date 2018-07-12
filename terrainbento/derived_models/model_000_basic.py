#! /usr/env/python
"""``terrainbento`` Model ``Basic`` program.

Erosion model program using linear diffusion, stream power, and discharge
proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class Basic(ErosionModel):
    """Model ``Basic`` program.

    Model ``Basic`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{w}A^{m}S^{n} + D\\nabla^2 \eta

    where :math:`A` is the local drainage area and :math:`S` is the local slope.
    Refer to the ``terrainbento`` manuscript Table XX (URL here) for parameter
    symbols, names, and dimensions.

    Model ``Basic`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class. Depending on the values of :math:`K_{w}`, :math:`D`, :math:`m`
    and, :math:`n` this model program can be used to run the following three
    ``terrainbento`` numerical models:

    1) Model **Basic**:
    +------------------+----------------------------------+-----------------+
    | Parameter Symbol | Input File Parameter Name        | Value           |
    +==================+==================================+=================+
    |:math:`m`         | ``m_sp``                         | 0.5             |
    +------------------+----------------------------------+-----------------+
    |:math:`n`         | ``n_sp``                         | 1               |
    +------------------+----------------------------------+-----------------+
    |:math:`K`         | ``water_erodability``            | user specified  |
    +------------------+----------------------------------+-----------------+
    |:math:`D`         | ``regolith_transport_parameter`` | user specified  |
    +------------------+----------------------------------+-----------------+

    2) Model **BasicSs**:
    +------------------+------------------------------------------+-----------------+
    | Parameter Symbol | Input File Parameter Name                | Value           |
    +==================+==========================================+=================+
    |:math:`m`         | ``m_sp``                                 | 1/3             |
    +------------------+------------------------------------------+-----------------+
    |:math:`n`         | ``n_sp``                                 | 2/3             |
    +------------------+------------------------------------------+-----------------+
    |:math:`K_{ss}`    | ``water_erodability~shear_stress``       | user specified  |
    +------------------+------------------------------------------+-----------------+
    |:math:`D`         | ``regolith_transport_parameter``         | user specified  |
    +------------------+------------------------------------------+-----------------+

    3) Model **BasicVm**:

    +------------------+------------------------------------------+-----------------+
    | Parameter Symbol | Input File Parameter Name                | Value           |
    +==================+==========================================+=================+
    |:math:`m`         | ``m_sp``                                 | user specified  |
    +------------------+------------------------------------------+-----------------+
    |:math:`n`         | ``n_sp``                                 | 1               |
    +------------------+------------------------------------------+-----------------+
    |:math:`K_{ss}`    | ``water_erodability~shear_stress``       | user specified  |
    +------------------+------------------------------------------+-----------------+
    |:math:`D`         | ``regolith_transport_parameter``         | user specified  |
    +------------------+------------------------------------------+-----------------+

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
        Basic : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``Basic``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import Basic

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
        ...           'n_sp': 1.0}

        Construct the model.

        >>> model = Basic(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(Basic, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters:
        K_sp = self.get_parameter_from_exponent("water_erodability", raise_error=False)
        K_ss = self.get_parameter_from_exponent(
            "water_erodability~shear_stress", raise_error=False
        )
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError(
                (
                    "Model 000: A parameter for both "
                    "water_erodability and "
                    "water_erodability~shear_stress has been provided. "
                    " Only one of these may be provided."
                )
            )
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (
                    self._length_factor ** (1. / 3.)
                ) * K_ss  # K_ss has units Length^(1/3) per Time
        else:
            raise ValueError(
                (
                    "water_erodability or "
                    "water_erodability~shear_stress must be provided."
                )
            )

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid, K_sp=self.K, m_sp=self.params["m_sp"], n_sp=self.params["n_sp"]
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, dt):
        """Advance model ``Basic`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the ``ErosionModel`` base class function
           **finalize__run_one_step**. This function updates all BoundaryHandlers
           by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """
        # Direct and accumulate flow
        self.flow_accumulator.run_one_step()

        # Get IDs of flooded nodes, if any.
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # If a PrecipChanger is being used, update the eroder's K value.
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodibility_adjustment_factor()
            )

        # Do some water erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main(): #pragma: no cover
    """Execute model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print(
            (
                "To run a terrainbento model from the command line you must "
                "include input file name on command line"
            )
        )
        sys.exit(1)

    model = Basic(input_file=infile)
    model.run()


if __name__ == "__main__":
    main()
