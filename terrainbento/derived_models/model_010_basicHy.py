# coding: utf8
#! /usr/env/python
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
    """**BasicHy** model program.

    **BasicHy** is a model program that evolves a topographic surface described
    by :math:`\eta` with the following governing equation:


    .. math::

        \\frac{\partial \eta}{\partial t} = -\left(KA^{m}S^{n} - \omega_c\left(1-e^{-KA^{m}S^{n}/\omega_c}\\right)\\right) + \\frac{V\\frac{Q_s}{Q}}{\left(1-\phi\\right)} + D\\nabla^2 \eta


    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`m` and :math:`n` are the drainage area and slope exponent parameters,
    :math:`K` is the erodability by water, :math:`\omega_c` is the critical
    stream power needed for erosion to occur, :math:`V` is effective sediment
    settling velocity, :math:`Q_s` is volumetric sediment flux, :math:`Q` is
    volumetric water discharge, :math:`\phi` is sediment porosity, :math:`D` is
    the regolith transport efficiency, :math:`H` is soil depth, and :math:`H_*`
    is the bedrock roughness length scale.

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

    A value for the parameter ``solver`` can also be used to indicate if the
    default internal timestepping is used for the **ErosionDeposition**
    component or if an adaptive internal timestep is used. Refer to the
    **ErosionDeposition** documentation for details.

    Refer to the terrainbento manuscript Table XX (URL here) for full list of
    parameter symbols, names, and dimensions.


    XXX todo add threshold.
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
        BasicHy : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicHy**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicHy

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
        ...           'v_sc': 0.01,
        ...           'phi': 0,
        ...           'F_f': 0,
        ...           'solver': 'basic'}

        Construct the model.

        >>> model = BasicHy(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicHy, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self.get_parameter_from_exponent("water_erodability") * (
            self._length_factor ** (1. - (2. * self.m))
        )

        # Unit conversion for linear_diffusivity, with units L^2/T
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        # Normalized settling velocity (dimensionless)
        v_sc = self.get_parameter_from_exponent("v_sc")

        # Handle solver option
        solver = self.params.get("solver", "basic")

        # Instantiate a Space component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            phi=self.params["phi"],
            F_f=self.params["F_f"],
            v_s=v_sc,
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
        """Advance model **BasicHy** for one time-step of duration dt.

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
                self.K
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(
            dt,
            flooded_nodes=flooded,
            dynamic_dt=True,
            flow_director=self.flow_accumulator.flow_director,
        )

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


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
