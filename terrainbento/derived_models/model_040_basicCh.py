#! /usr/env/python
""" ``terrainbento`` Model ``BasicCh`` program.

Erosion model program using cubic diffusion, basic stream
power, and discharge proportional to drainage area.


Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `TaylorNonLinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.taylor_nonlinear_hillslope_flux.html>`_

"""

import sys
import numpy as np

from landlab.components import FastscapeEroder, TaylorNonLinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicCh(ErosionModel):
    """

    Model ``BasicCh`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:


    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{w}A^{m}S^{n} + \\nabla^2 q_h


    where


    .. math::

        q_h = -DS \left[ 1 + \left( \\frac{S}{S_c} \\right)^2 +  \left( \\frac{S}{S_c} \\right)^4 + ... \left( \\frac{S}{S_c} \\right)^{2(N-1)} \\right]


    where :math:`S_c` is the critical slope, :math:`A` is the local drainage
    area and :math:`S` is the local slope. :math:`N` is the number of terms in
    the Taylor Expansion and is set at 11. Refer to the ``terrainbento``
    manuscript Table XX (URL here) for parameter symbols, names, and dimensions.

    Model ``BasicCh`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class.

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
    |:math:`S_c`       | ``critical_slope``               | user specified  |
    +------------------+----------------------------------+-----------------+


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
        BasicCh : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``BasicCh``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicCh

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'critical_slope': 0.2,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        Construct the model.

        >>> model = BasicCh(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicCh, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K_sp,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a NonLinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=self.params["critical_slope"],
            nterms=11,
        )

    def run_one_step(self, dt):
        """Advance model ``BasicCh`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by nonlinear diffusion.

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

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K_sp
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(
            dt, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

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

    cdsp = BasicCh(input_file=infile)
    cdsp.run()


if __name__ == "__main__":
    main()
