#! /usr/env/python
"""``terrainbento`` Model ``BasicHy`` program.

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
    """Model ``BasicHy`` program.

    Model ``BasicHy`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -\left(K_{w}A^{m}S^{n} - \\ 
        \omega_c\left(1-e^{-K_{w}A^{m}S^{n}/\omega_c}\\right)\\right) + \\
        \\frac{V\\frac{Q_s}{Q}}{\left(1-\phi\\right)} + D\\nabla^2 \eta

    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`H` is soil depth, :math:`H_*` is the bedrock roughnes length scale,
    :math:`\omega_c` is the critical stream power needed for erosion to occur,
    :math:`V` is effective sediment settling velocity, :math:`Q_s` is
    volumetric sediment flux, :math:`Q` is volumetric water discharge, and 
    :math:`\phi` is sediment porosity. Refer to the ``terrainbento`` 
    manuscript Table XX (URL here) for parameter symbols, names, and 
    dimensions.

    Model ``BasicHy`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class. Depending on the value of :math:`\omega_c`, this model program can 
    be used to run the following two ``terrainbento`` numerical models:

    1) Model ``BasicHy``: Here there is no erosion threshold, i.e. 
    :math:`\omega_c=0`.

    2) Model ``BasicHyTh``: This model is identical to Model BasicHy except
    that the erosion threshold :math:`\omega_c` is nonzero.
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
        of model ``BasicHy``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

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
        K_sp = self.get_parameter_from_exponent("water_erodability", raise_error=False)
        K_ss = self.get_parameter_from_exponent(
            "water_erodability~shear_stress", raise_error=False
        )

        # check that a stream power and a shear stress parameter have not both been given
        if K_sp != None and K_ss != None:
            raise ValueError(
                "A parameter for both K_rock_sp and K_rock_ss has been"
                "provided. Only one of these may be provided"
            )
        elif K_sp != None or K_ss != None:
            if K_sp != None:
                self.K = K_sp
            else:
                self.K = (
                    self._length_factor ** (1. / 3.)
                ) * K_ss  # K_ss has units Lengtg^(1/3) per Time
        else:
            raise ValueError("A value for K_rock_sp or K_rock_ss  must be provided.")

        # Unit conversion for linear_diffusivity, with units L^2/T
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        # Normalized settling velocity (dimensionless)
        v_sc = self.get_parameter_from_exponent("v_sc")

        # make area_field and/or discharge_field depending on discharge_method
        #        area_field = self.grid.at_node['drainage_area']
        #        discharge_field = None

        # Handle solver option
        try:
            solver = self.params["solver"]
        except:
            solver = "original"

        # Instantiate a Space component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K,
            phi=self.params["phi"],
            F_f=self.params["F_f"],
            v_s=v_sc,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            method="simple_stream_power",
            discharge_method="drainage_area",
            area_field="drainage_area",
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, dt):
        """Advance model ``BasicHy`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
        not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
        so, uses it to modify the erodability by water.

        4. Calculates erosion and deposition by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the ``ErosionModel`` base class function
        **finalize__run_one_step**. This function updates all BoundaryHandlers
        by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """

        # Route flow
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
                ].get_erodibility_adjustment_factor()
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


def main(): #pragma: no cover
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
