#! /usr/env/python
"""``terrainbento`` Model ``BasicDdHy`` program.

Erosion model program using linear diffusion, stream-power-driven sediment 
erosion and mass conservation with a smoothed threshold that varies with 
incision depth, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `ErosionDeposition <http://landlab.readthedocs.io/en/release/landlab.components.erosion_deposition.html>`_
    4. `LinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.diffusion.html>`_
"""

import numpy as np

from landlab.components import LinearDiffuser, ErosionDeposition
from terrainbento.base_class import ErosionModel


class BasicDdHy(ErosionModel):
    """Model ``BasicDdHy`` program.

    Model ``BasicDdHy`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -\left(K_{w}A^{m}S^{n} - \\ 
        \omega_{ct}\left(1-e^{-K_{w}A^{m}S^{n}/\omega_{ct}}\\right)\\right) + \\
        \\frac{V\\frac{Q_s}{Q}}{\left(1-\phi\\right)} + D\\nabla^2 \eta

    where :math:`A` is the local drainage area, :math:`S` is the local slope, 
    :math:`H` is soil depth, :math:`H_*` is the bedrock roughnes length scale,
    :math:`V` is effective sediment settling velocity, :math:`Q_s` is
    volumetric sediment flux, :math:`Q` is volumetric water discharge, and 
    :math:`\phi` is sediment porosity. :math:`\omega_{ct}` is the critical 
    stream power needed for erosion to occur, which may change through time as 
    it increases with cumulative incision depth:
        
    .. math::
        
        \omega_{ct}\left(x,y,t\\right) = \mathrm{max}\left(\omega_c + \\
        b D_I\left(x, y, t\\right), \omega_c \\right)
            
    where :math:`\omega_c` is the threshold when no incision has taken place, 
    :math:`b` is the rate at which the threshold increases with incision depth,
    and :math:`D_I` is the cumulative incision depth at location 
    :math:`\left(x,y\\right)` and time :math:`t`.
    
    Refer to the ``terrainbento`` manuscript Table XX (URL here) for parameter 
    symbols, names, and dimensions.

    Model ``BasicDdHy`` inherits from the ``terrainbento`` ``ErosionModel`` 
    base class. 
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
        BasicDdHy : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``BasicDdHy``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicDdHy

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
        ...           'solver': 'basic',
        ...           'erosion__threshold': 0.01,
        ...           'thresh_change_per_depth': 0.01}

        Construct the model.

        >>> model = BasicDdHy(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0
        
        """

        # Call ErosionModel's init
        super(BasicDdHy, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2
        ) * self.get_parameter_from_exponent(  # L2/T
            "regolith_transport_parameter"
        )
        v_s = self.get_parameter_from_exponent("v_sc")  # unitless
        self.sp_crit = self._length_factor * self.get_parameter_from_exponent(  # L/T
            "erosion__threshold"
        )

        # Create a field for the (initial) erosion threshold
        self.threshold = self.grid.add_zeros("node", "erosion__threshold")
        self.threshold[:] = self.sp_crit  # starting value

        # Handle solver option
        try:
            solver = self.params["solver"]
        except:
            solver = "basic"

        # Instantiate an ErosionDeposition component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K_sp,
            F_f=self.params["F_f"],
            phi=self.params["phi"],
            v_s=v_s,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            sp_crit="erosion__threshold",
            method="threshold_stream_power",
            discharge_method="drainage_area",
            area_field="drainage_area",
            solver=solver,
        )

        # Get the parameter for rate of threshold increase with erosion depth
        self.thresh_change_per_depth = self.params["thresh_change_per_depth"]

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, dt):
        """Advance model ``BasicDdHy`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
        not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
        so, uses it to modify the erodability by water.

        4. Calculates threshold-modified erosion and deposition by water.

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

        # Calculate cumulative erosion and update threshold
        cum_ero = self.grid.at_node["cumulative_erosion__depth"]
        cum_ero[:] = self.z - self.grid.at_node["initial_topographic__elevation"]
        self.threshold[:] = self.sp_crit - (self.thresh_change_per_depth * cum_ero)
        self.threshold[self.threshold < self.sp_crit] = self.sp_crit

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
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main(): #pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    em = BasicDdHy(input_file=infile)
    em.run()


if __name__ == "__main__":
    main()
