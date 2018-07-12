#! /usr/env/python
"""``terrainbento`` Model ``BasicHySa`` program.

Erosion model program using exponential weathering, soil-depth-dependent
linear diffusion, stream-power-driven sediment erosion, mass conservation, and
bedrock erosion, and discharge proportional to drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `Space <http://landlab.readthedocs.io/en/release/landlab.components.space.html>`_
    4. `DepthDependentDiffuser <http://http://landlab.readthedocs.io/en/release/landlab.components.depth_dependent_diffusion.html>`_
    5. `ExponentialWeatherer <http://http://landlab.readthedocs.io/en/release/landlab.components.weathering.html>`_
"""

import numpy as np

from landlab.components import Space, DepthDependentDiffuser, ExponentialWeatherer
from terrainbento.base_class import ErosionModel


class BasicHySa(ErosionModel):
    """Model ``BasicHySa`` program.

    Model ``BasicHySa`` is a model program that evolves a topographic surface
    described by :math:`\eta` with the following governing equation:

    .. math::

        \\frac{\partial \eta}{\partial t} = -K_{r}A^{m}S^{n}\left(e^{-H/H_*}\\right) \\
        -K_{w}A^{m}S^{n}\left(1-e^{-H/H_*}\\right) \\
        + \\frac{V\\frac{Q_s}{Q}}{\left(1-\phi\\right)} + \\nabla q_s

    where :math:`K_r` and :math:`K_s` are rock and sediment erodibility
    respectively, :math:`A` is the local drainage area, :math:`S` is the local
    slope, :math:`H` is soil depth, :math:`H_*` is the bedrock roughnes length
    scale, :math:`\omega_c` is the critical stream power needed for erosion to
    occur, :math:`V` is effective sediment settling velocity, :math:`Q_s` is
    volumetric fluvial sediment flux, :math:`Q` is volumetric water discharge,
    and :math:`\phi` is sediment porosity. Hillslope sediment flux per unit
    width :math:`q_s` is given by:

    .. math::
        q_s=-DS\left(1-e^{-H/H_0}\\right)

    where :math:`D` is soil diffusivity and :math:`H_0` is the soil transport
    depth scale.

    Refer to the ``terrainbento`` manuscript Table XX (URL here) for parameter
    symbols, names, and dimensions.

    Model ``BasicHySa`` inherits from the ``terrainbento`` ``ErosionModel``
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
        BasicHySa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``BasicHySa``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicHySa

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'K_rock_sp': 0.001,
        ...           'K_sed_sp': 0.001,
        ...           'sp_crit_br': 0,
        ...           'sp_crit_sed': 0,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'v_sc': 0.01,
        ...           'phi': 0,
        ...           'F_f': 0,
        ...           'H_star': 0.1,
        ...           'solver': 'basic',
        ...           'soil_transport_decay_depth': 1,
        ...           'soil_production__maximum_rate': 0.0001,
        ...           'soil_production__decay_depth': 0.5}

        Construct the model.

        >>> model = BasicHySa(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicHySa, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        self.K_br = self.get_parameter_from_exponent("K_rock_sp")
        self.K_sed = self.get_parameter_from_exponent("K_sed_sp")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        v_sc = self.get_parameter_from_exponent(
            "v_sc"
        )  # normalized settling velocity. Unitless.

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor) * self.params[
                "initial_soil_thickness"
            ]  # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor) * self.params[
            "soil_transport_decay_depth"
        ]  # has units length
        max_soil_production_rate = (self._length_factor) * self.params[
            "soil_production__maximum_rate"
        ]  # has units length per time
        soil_production_decay_depth = (self._length_factor) * self.params[
            "soil_production__decay_depth"
        ]  # has units length

        # set methods and fields. K's and sp_crits need to be field names
        method = self.params.get("space_method", "simple_stream_power")
        discharge_method = self.params.get("discharge_method", "discharge_field")
        area_field = self.params.get("area_field", None)
        discharge_field = self.params.get("discharge_field", "surface_water__discharge")
        K_noise_scale = self.params.get("K_noise_scale", 0)

        # Handle solver option
        solver = self.params.get("solver", "basic")

        # Instantiate a SPACE component
        self.eroder = Space(
            self.grid,
            K_sed=self.K_sed,
            K_br=self.K_br,
            sp_crit_br=self.params["sp_crit_br"],
            sp_crit_sed=self.params["sp_crit_sed"],
            F_f=self.params["F_f"],
            phi=self.params["phi"],
            H_star=self.params["H_star"],
            v_s=v_sc,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            method=method,
            discharge_method=discharge_method,
            area_field=area_field,
            discharge_field=discharge_field,
            solver=solver,
            K_noise_scale=K_noise_scale,
        )

        # SPACE checks for and creates bedrock elevation and soil depth
        # grid fields when instantiated, so no need to do that here in
        # the model.

        # Get soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.at_node["soil__depth"]

        # Get bedrock elevation field
        bedrock_elev = self.grid.at_node["bedrock__elevation"]

        # Set soil thickness and bedrock elevation
        try:
            initial_soil_thickness = self.params["initial_soil_thickness"]
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=max_soil_production_rate,
            soil_production__decay_depth=soil_production_decay_depth,
        )

        self.grid.at_node["soil__depth"][:] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["bedrock__elevation"]
        )

    def run_one_step(self, dt):
        """Advance model ``BasicHySa`` for one time-step of duration dt.

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
            erode_factor = self.boundary_handler[
                "PrecipChanger"
            ].get_erodibility_adjustment_factor()
            self.eroder.K_sed = self.K_sed * erode_factor
            self.eroder.K_br = self.K_br * erode_factor

        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)

        # Check stability
        self.check_stability()

    def check_stability(self):
        """Check model stability and exit if unstable."""
        fields = self.grid.at_node.keys()
        for f in fields:
            if np.any(np.isnan(self.grid.at_node[f])) or np.any(
                np.isinf(self.grid.at_node[f])
            ):

                # model is unstable, write message and exit.
                with open("model_failed.txt", "w") as f:
                    f.write("This model run became unstable\n")

                raise SystemExit("Model became unstable")


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    hysa = BasicHySa(input_file=infile)
    hysa.run()


if __name__ == "__main__":
    main()
