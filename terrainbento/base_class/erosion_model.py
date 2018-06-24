#! /usr/env/python
"""Base class for common functions of all ``terrainbento`` erosion models.

The ``ErosionModel`` is a base class that contains all of the functionality
shared by the ``terrainbento`` models.


Input File or Dictionary Parameters
-----------------------------------
The following are parameters found in the parameters input file or dictionary.
Depending on how the model is initialized, some of them are optional or not
used.

Required Parameters
^^^^^^^^^^^^^^^^^^^
The required parameters control how long a model will run, the duration of a
model timestep, and the interval at which output is written.

run_duration : float
    Duration of entire model run.
dt : float
    Increment of time at which the model is run (i.e., time-step duration).
output_interval : float
    Increment of model time at which model output is written.


Grid Setup Parameters
^^^^^^^^^^^^^^^^^^^^^
This set of parameters controls what kind of model grid is created. Two primary
options exist: the creation of a model grid based on elevations provided by an
input DEM, and the creation of a synthetic model domain. In this latter option,
either a  ``RasterModelGrid`` or  ``HexModelGrid`` of synthetic terrain is
possible.

If neither of the two following parameters is specified, a synthetic
``RasterModelGrid`` will be created. If parameters associated with setting up
a synthetic ``RasterModelGrid`` are not provided, default values will be used
for grid size, initial topography, and boundary conditions.

DEM_filename : str, optional
    File path to either an ESRII ASCII or netCDF file. Either  ``'DEM_filename'``
    or ``'model_grid'`` must be specified.
model_grid : str, optional
    Either ``'RasterModelGrid'`` or ``'HexModelGrid'``.

Note that if both ``'DEM_filename'`` and ``'model_grid'`` are specified,
an error will be raised.

Parameters that control creation of a synthetic HexModelGrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These parameters control the size, shape, and model boundary conditions of a
synthetic HexModelGrid. They are only used if ``model_grid == 'HexModelGrid'``.

number_of_node_rows : int, optional
    Number of rows of nodes in the left column. Default is 8.
number_of_node_columns : int, optional
    Number of nodes on the first row. Default is 5.
node_spacing : float, optional
    Node spacing. Default is 10.0.
orientation : str, optional
    Either 'horizontal' (default) or 'vertical'.
shape : str, optional
    Controls the shape of the bounding hull, i.e., are the nodes
    arranged in a hexagon, or a rectangle? Either 'hex' (default) or
    'rect'.
reorient_links : bool, optional
    Whether or not to re-orient all links to point between -45 deg
    and +135 deg clockwise from "north" (i.e., along y axis). Default
    value is True.
outlet_id : int, optional
    Node id for the watershed outlet. If not provided, the model will be
    set boundary conditions based on the following parameters.
boundary_closed : boolean, optional
    If ``True`` the model boundarys are closed boundaries. Default is
    ``False``.

Parameters that control creation of a synthetic RasterModelGrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These parameters control the size, shape, and model boundary conditions of a
synthetic ``RasterModelGrid``.  These parameters are used if neither
``DEM_filename`` nor ``'model_grid'`` is specified or if
`model_grid == 'RasterModelGrid'``.

number_of_node_rows : int, optional
    Number of node rows. Default is 4.
number_of_node_columns : int, optional
    Number of node columns. Default is 5.
node_spacing : float, optional
    Row and column node spacing. Default is 1.0.
outlet_id : int, optional
    Node id for the watershed outlet. If not
    provided, the model will set boundary conditions
    based on the following parameters.
east_boundary_closed : boolean
    If ``True`` right-edge nodes are closed boundaries. Default is ``False``.
north_boundary_closed : boolean
    If ``True`` top-edge nodes are closed boundaries. Default is ``False``.
west_boundary_closed : boolean
    If ``True`` left-edge nodes are closed boundaries. Default is ``False``.
south_boundary_closed : boolean
    If ``True`` bottom-edge nodes are closed boundaries. Default is ``False``.

Parameters that control creation of synthetic topography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These parameters create synthetic initial topgraphy in either a
``RasterModelGrid`` or ``HexModelGrid``. They are used if ``DEM_filename`` is
not specified.

initial_elevation : float, optional
    Default value is 0.
random_seed : int, optional
    Default value is 0.
add_random_noise : boolean, optional
    Default value is True.
initial_noise_std : float, optional
    Standard deviation of zero-mean, normally distributed random perturbations
    to initial node elevations. Default value is 0.
add_noise_to_all_nodes : bool, optional
    When False, noise is added to core nodes only. Default value is False.

Parameters that control grid boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``terrainbento`` provides the ability for an arbitrary number of boundary
condition handler classes to operate on the model grid each time step in order
to handle time-variable boundary conditions such as: changing a watershed outlet
elevation, modifying precipitation parameters through time, or simulating
external drainage capture.

Boundary condition handlers are styled after Landlab components. ``terrainbento``
presently has four built-in boundary condition handlers, and supports the use
of the Landlab NormalFault component as a fifth. Over time the developers
anticipate extending the boundary handler library to include other Landlab
components and other options within ``terrainbento``. If these present
capabilities do not fit your needs, we recommend that you make an issue
describing the functionality you would like to use in your work.

BoundaryHandlers : str or list of str, optional
    Strings containing the names of classes used to handle boundary conditions.
    Valid options are currently: 'NormalFault', 'PrecipChanger',
    'CaptureNodeBaselevelHandler', 'NotCoreNodeBaselevelHandler', and
    'SingleNodeBaselevelHandler'. These BoundaryHandlers are instantiated with
    the entire parameter set unless there is an entry in the parameter
    dictionary with the name of the boundary handler that contains its own
    parameter dictionary. If this is the case, the handler-specific dictionary
    is passed to instantiate the boundary handler.

Parameters that control units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These courtesy options exist to support the case in which a model must be run
in one type of units (e.g. feet) but the scientific literature  provides
information about parameter values in a different unit (e.g. meters). If both
are set to ``True`` a ``ValueError`` will be raised.

meters_to_feet : boolean, optional
    Default value is False.
feet_to_meters : boolean, optional
    Default value is False.

Parameters that control surface hydrology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``terrainbento`` uses the Landlab FlowAccumulator component to manage surface
hydrology. These parameters control options associated with this component.

flow_director : str, optional
    String name of a Landlab FlowDirector. All options that the Landlab
    FlowAccumulator is compatible with are permitted. Default is
    'FlowDirectorSteepest'.
depression_finder : str, optional
    String name of a Landlab depression finder. Default is no depression finder.

Parameters that control output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In addition to the required parameter ``output_interval``, the following
parameters control when and how output is written.

save_first_timestep : bool, optional
    Indicates whether model output should be saved at time zero.  Default is
    True.
output_filename : str, optional
    String prefix for names of output netCDF files. Default is
    ``'terrainbento_output'``.

Note also that the **run** method takes as a parameter ``output_fields``, which
is a list of model grid fields to write as output.
"""

import sys
import time as tm
import numpy as np

from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab import load_params
from landlab.io.netcdf import write_raster_netcdf
from landlab.graph import Graph

from landlab import Component, CLOSED_BOUNDARY
from landlab.components import FlowAccumulator, NormalFault

from terrainbento.boundary_condition_handlers import (
    PrecipChanger,
    CaptureNodeBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    SingleNodeBaselevelHandler,
)

_SUPPORTED_BOUNDARY_HANDLERS = [
    "NormalFault",
    "PrecipChanger",
    "CaptureNodeBaselevelHandler",
    "NotCoreNodeBaselevelHandler",
    "SingleNodeBaselevelHandler",
]

_HANDLER_METHODS = {
    "NormalFault": NormalFault,
    "PrecipChanger": PrecipChanger,
    "CaptureNodeBaselevelHandler": CaptureNodeBaselevelHandler,
    "NotCoreNodeBaselevelHandler": NotCoreNodeBaselevelHandler,
    "SingleNodeBaselevelHandler": SingleNodeBaselevelHandler,
}


class ErosionModel(object):

    """Base class providing common functionality for ``terrainbento`` models.

    An ``ErosionModel`` is the skeleton for the models of terrain evolution in
    ``terrainbento``. It can be initialized with an input DEM, or parameters
    used for creation of a new ``RasterModelGrid`` or ``HexModelGrid``.

    This is a base class that does not implement any processes, but rather
    simply handles I/O and setup. Derived classes are meant to include
    Landlab components to model actual erosion processes.

    It is expected that a derived model will define an **__init__** and a
    **run_one_step** method. If desired, the derived model can overwrite the
    existing **run_for**, **run**, and **finalize** methods.
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
            Dictionary containing the input file. One of input_file or params
            is required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions,
            optional Classes or functions used to write incremental output
            (e.g. make a diagnostic plot).

        Returns
        -------
        ErosionModel : object

        Examples
        --------
        We recommend that you look at the ``terrainbento`` tutorials for
        examples of usage.
        """
        #######################################################################
        # get parameters
        #######################################################################
        # Import input file or parameter dictionary, checking that at least
        # one but not both were supplied.
        if input_file is None and params is None:
            raise ValueError(
                (
                    "ErosionModel requires one of `input_file` or "
                    "`params` dictionary but neither were supplied."
                )
            )
        elif input_file is not None and params is not None:
            raise ValueError(
                (
                    "ErosionModel requires one of `input_file` or "
                    "`params` dictionary but both were supplied."
                )
            )
        else:
            # parameter dictionary
            if input_file is None:
                self.params = params
            # read from file.
            else:
                self.params = load_params(input_file)

        # ensure required values are provided
        for req in ["dt", "output_interval", "run_duration"]:
            if req in self.params:
                try:
                    val = float(self.params[req])
                except ValueError:
                    msg = (
                        "Required parameter {0} is not compatible with type float.".format(
                            req
                        ),
                    )
                    raise ValueError(msg)
            else:
                msg = ("Required parameter {0} was not provided.".format(req),)

                raise ValueError(msg)

        # identify if initial conditions should be saved.
        # default behavior is to not save the first timestep
        self.save_first_timestep = self.params.get("save_first_timestep", True)
        self._out_file_name = self.params.get("output_filename", "terrainbento_output")
        # instantiate model time.
        self._model_time = 0.

        # instantiate container for computational timestep:
        self._compute_time = [tm.time()]

        ###################################################################
        # create topography
        ###################################################################

        # Read the topography data and create a grid
        # first, check to make sure both DEM and node-rows are not both
        # specified.
        if (self.params.get("number_of_node_rows") is not None) and (
            self.params.get("DEM_filename") is not None
        ):
            raise ValueError(
                "Both a DEM filename and number_of_node_rows have " "been specified."
            )

        if "DEM_filename" in self.params:
            self._starting_topography = "inputDEM"
            (self.grid, self.z) = self.read_topography(self.params["DEM_filename"])
            self.opt_watershed = True
        else:
            # this routine will set self.opt_watershed internally
            if self.params.get("model_grid", "RasterModelGrid") == "HexModelGrid":
                self._starting_topography = "HexModelGrid"
                self.setup_hexagonal_grid()
            else:
                self._starting_topography = "RasterModelGrid"
                self.setup_raster_grid()

        # Set DEM boundaries
        if self.opt_watershed:
            if "outlet_id" in self.params:
                self.outlet_node = self.params["outlet_id"]
                self.grid.set_watershed_boundary_condition_outlet_id(
                    self.outlet_node, self.z, nodata_value=-9999
                )
            else:
                self.outlet_node = self.grid.set_watershed_boundary_condition(
                    self.z, nodata_value=-9999, return_outlet_id=True
                )[0]

        # Add fields for initial topography and cumulative erosion depth
        z0 = self.grid.add_zeros("node", "initial_topographic__elevation")
        z0[:] = self.z  # keep a copy of starting elevation
        self.grid.add_zeros("node", "cumulative_erosion__depth")

        # identify which nodes are data nodes:
        self.data_nodes = self.grid.at_node["topographic__elevation"] != -9999.

        ###################################################################
        # instantiate flow direction and accumulation
        ###################################################################
        # get flow direction, and depression finding options
        self.flow_director = self.params.get("flow_director", "FlowDirectorSteepest")
        if (self.flow_director == "Steepest") or (self.flow_director == "D4"):
            self.flow_director = "FlowDirectorSteepest"
        self.depression_finder = self.params.get("depression_finder", None)

        # Instantiate a FlowAccumulator, if DepressionFinder is provided
        # AND director = Steepest, then we need routing to be D4,
        # otherwise, just passing params should be sufficient.
        if (self.depression_finder is not None) and (
            self.flow_director == "FlowDirectorSteepest"
        ):
            self.flow_accumulator = FlowAccumulator(
                self.grid, routing="D4", **self.params
            )
        else:
            self.flow_accumulator = FlowAccumulator(self.grid, **self.params)

        ###################################################################
        # get internal length scale adjustement
        ###################################################################
        feet_to_meters = self.params.get("feet_to_meters", False)
        meters_to_feet = self.params.get("meters_to_feet", False)
        if feet_to_meters and meters_to_feet:
            raise ValueError(
                'Both "feet_to_meters" and "meters_to_feet" are'
                "set as True. This is not realistic."
            )
        else:
            if feet_to_meters:
                self._length_factor = 1.0 / 3.28084
            elif meters_to_feet:
                self._length_factor = 3.28084
            else:
                self._length_factor = 1.0
        self.params["length_factor"] = self._length_factor

        ###################################################################
        # Boundary Conditions
        ###################################################################
        self.boundary_handler = {}
        if "BoundaryHandlers" in self.params:
            BoundaryHandlers = self.params["BoundaryHandlers"]

        if BoundaryHandlers is None:
            pass
        else:
            if isinstance(BoundaryHandlers, list):
                for comp in BoundaryHandlers:
                    self.setup_boundary_handler(comp)
            else:
                self.setup_boundary_handler(BoundaryHandlers)

        ###################################################################
        # Output Writers
        ###################################################################
        self.output_writers = {"class": {}, "function": []}
        if OutputWriters is None:
            pass
        else:
            if isinstance(OutputWriters, list):
                for comp in OutputWriters:
                    self.setup_output_writer(comp)
            else:
                self.setup_output_writer(OutputWriters)

    @property
    def model_time(self):
        """Return current time of model integration in model time units."""
        return self._model_time

    def setup_boundary_handler(self, handler):
        """ Setup BoundaryHandlers for use by a ``terrainbento`` model.

        A boundary condition handler is a class with a run_one_step method that
        takes the parameter ``dt``. Permitted boundary condition handlers
        include the Landlab Component ``NormalFault`` as well as the following
        options from ``terrainbento``: ``PrecipChanger``,
        ``CaptureNodeBaselevelHandler``, ``NotCoreNodeBaselevelHandler``,
        ``SingleNodeBaselevelHandler``.

        Parameters
        ----------
        handler : str or object
            Name of instance of a supported boundary condition handler.
        """
        try:  # if handler is an uninstantiated component
            name = handler._name

            if isinstance(handler, Component):
                raise ValueError(
                    (
                        "Object passed to terrainbento is instantiated "
                        ". This is not permitted."
                    )
                )
        except AttributeError:
            try:  # if handler is a string
                name = handler
                handler = _HANDLER_METHODS[name]
            except KeyError:
                raise ValueError(
                    (
                        "Object passed to terrainbento init is not a "
                        "valid Boundary Handler."
                    )
                )

        if name in _SUPPORTED_BOUNDARY_HANDLERS:

            # if unique paramters for the boundary condition handler have
            # been passed, use them.
            if name in self.params:
                handler_params = self.params[name]
                handler_params["length_factor"] = self._length_factor

            # otherwise pass all parameters
            else:
                handler_params = self.params

            # Instantiate handler
            self.boundary_handler[name] = handler(self.grid, **handler_params)

        # Raise an error if the handler is not supported.
        else:
            raise ValueError(
                (
                    "Only supported boundary condition handlers are "
                    "permitted. These include:"
                    "\n".join(_SUPPORTED_BOUNDARY_HANDLERS)
                )
            )

    def setup_output_writer(self, writer):
        """Setup OutputWriter for use by a ``terrainbento`` model.

        An OutputWriter can be either a function or a class designed to create
        output, calculate a loss function, or do some other task that is not
        inherent to running a ``terrainbento`` model but is desired by the
         user. An example might be making a plot of topography while the model
        is running. ``terrainbento`` saves output to NetCDF format at each
        interval defined by the parameter ``'output_interval'``.

        If a class, an OutputWriter will be instantiated with only one passed
        argument: the entire model object. The class is expected to have a bound
        function called **run_one_step** which is run with no arguments each time
        output is written. If a function, the OutputWriter will be run at each
        time output is written with one passed argument: the entire model
        object.

        Parameters
        ----------
        writer : function or class
            An OutputWriter function or class
        """
        if isinstance(writer, object):
            name = writer.__name__
            self.output_writers["class"][name] = writer(self)
        else:
            self.output_writers["function"].append(writer)

    def setup_hexagonal_grid(self):
        """Create hexagonal grid based on input parameters.

        This method will be called if the value of the input parameter
        ``'DEM_filename'`` does not exist, and if the value of the input parameter
        ``'model_grid'`` is set to `'HexModelGrid'`. Input parameters are not
        passed explicitly, but are expected to be located in the model attribute
        ``params``.

        Parameters
        ----------
        number_of_node_rows : int, optional
            Number of rows of nodes in the left column. Default is 8.
        number_of_node_columns : int, optional
            Number of nodes on the first row. Default is 5.
        node_spacing : float, optional
            Node spacing. Default is 10.0.
        orientation : str, optional
            Either 'horizontal' (default) or 'vertical'.
        shape : str, optional
            Controls the shape of the bounding hull, i.e., are the nodes
            arranged in a hexagon, or a rectangle? Either ``'hex'`` (default) or
            ``'rect'``.
        reorient_links, bool, optional
            Whether or not to re-orient all links to point between -45 deg
            and +135 deg clockwise from "north" (i.e., along y axis). Default
            value is True.
        outlet_id : int, optional
            Node id for the watershed outlet. If not provided, the model will be
            set boundary conditions based on the following parameters.
        boundary_closed : boolean, optional
            If ``True`` the model boundarys are closed boundaries. Default is
            ``False``.

        Examples
        --------
        >>> from landlab import HexModelGrid
        >>> from terrainbento import ErosionModel
        >>> params = {'model_grid' : 'HexModelGrid',
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'dt': 1, 'output_interval': 2., 'run_duration': 10.}

        >>> em = ErosionModel(params=params)
        >>> isinstance(em.grid, HexModelGrid)
        True
        >>> em.grid.x_of_node
        array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  -5.,   5.,
                15.,  25.,  35.,  45.,  55.,  65.,  75.,  85., -10.,   0.,  10.,
                20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., -15.,  -5.,   5.,
                15.,  25.,  35.,  45.,  55.,  65.,  75.,  85.,  95., -10.,   0.,
                10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90.,  -5.,   5.,
                15.,  25.,  35.,  45.,  55.,  65.,  75.,  85.])
        >>> em.grid.y_of_node
        array([  0.        ,   0.        ,   0.        ,   0.        ,
                 0.        ,   0.        ,   0.        ,   0.        ,
                 0.        ,   8.66025404,   8.66025404,   8.66025404,
                 8.66025404,   8.66025404,   8.66025404,   8.66025404,
                 8.66025404,   8.66025404,   8.66025404,  17.32050808,
                17.32050808,  17.32050808,  17.32050808,  17.32050808,
                17.32050808,  17.32050808,  17.32050808,  17.32050808,
                17.32050808,  17.32050808,  25.98076211,  25.98076211,
                25.98076211,  25.98076211,  25.98076211,  25.98076211,
                25.98076211,  25.98076211,  25.98076211,  25.98076211,
                25.98076211,  25.98076211,  34.64101615,  34.64101615,
                34.64101615,  34.64101615,  34.64101615,  34.64101615,
                34.64101615,  34.64101615,  34.64101615,  34.64101615,
                34.64101615,  43.30127019,  43.30127019,  43.30127019,
                43.30127019,  43.30127019,  43.30127019,  43.30127019,
                43.30127019,  43.30127019,  43.30127019])
        """
        try:
            nr = self.params["number_of_node_rows"]
            nc = self.params["number_of_node_columns"]
            dx = self.params["node_spacing"]

        except KeyError:
            nr = 8
            nc = 5
            dx = 10.0
        orientation = self.params.get("orientation", "horizontal")
        shape = self.params.get("shape", "hex")
        reorient_links = self.params.get("reorient_links", True)

        # Create grid
        from landlab import HexModelGrid

        self.grid = HexModelGrid(
            nr,
            nc,
            dx,
            shape=shape,
            orientation=orientation,
            reorient_links=reorient_links,
        )

        # Create and initialize elevation field
        self._create_synthetic_topography()

        # Set boundary conditions
        self._setup_synthetic_boundary_conditions()

    def setup_raster_grid(self):
        """Create raster grid based on input parameters.

        This method will be called if the value of the input parameter
        ``'DEM_filename'`` does not exist, and if the value of the input parameter
        ``'model_grid'`` is set to ``'RasterModelGrid'``. Input parameters are not
        passed explicitly, but are expected to be located in the model attribute
        ``params``.

        Parameters
        ----------
        number_of_node_rows : int, optional
            Number of node rows. Default is 4.
        number_of_node_columns : int, optional
            Number of node columns. Default is 5.
        node_spacing : float, optional
            Row and column node spacing. Default is 1.0.
        outlet_id : int, optional
            Node id for the watershed outlet. If not
            provided, the model will set boundary conditions
            based on the following parameters.
        east_boundary_closed : boolean
            If ``True`` right-edge nodes are closed boundaries. Default is ``False``.
        north_boundary_closed : boolean
            If ``True`` top-edge nodes are closed boundaries. Default is ``False``.
        west_boundary_closed : boolean
            If ``True`` left-edge nodes are closed boundaries. Default is ``False``.
        south_boundary_closed : boolean
            If ``True`` bottom-edge nodes are closed boundaries. Default is ``False``.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> params = { 'number_of_node_rows' : 6,
        ...            'number_of_node_columns' : 9,
        ...            'node_spacing' : 10.0,
        ...            'dt': 1, 'output_interval': 2., 'run_duration': 10.}
        >>> from terrainbento import ErosionModel
        >>> em = ErosionModel(params=params)
        >>> em = ErosionModel(params=params)
        >>> isinstance(em.grid, RasterModelGrid)
        True
        >>> em.grid.x_of_node
        array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,   0.,  10.,
                20.,  30.,  40.,  50.,  60.,  70.,  80.,   0.,  10.,  20.,  30.,
                40.,  50.,  60.,  70.,  80.,   0.,  10.,  20.,  30.,  40.,  50.,
                60.,  70.,  80.,   0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,
                80.,   0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.])
        >>> em.grid.y_of_node
        array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,  10.,
                10.,  10.,  10.,  10.,  10.,  10.,  10.,  20.,  20.,  20.,  20.,
                20.,  20.,  20.,  20.,  20.,  30.,  30.,  30.,  30.,  30.,  30.,
                30.,  30.,  30.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,
                40.,  50.,  50.,  50.,  50.,  50.,  50.,  50.,  50.,  50.])
        """
        try:
            nr = self.params["number_of_node_rows"]
            nc = self.params["number_of_node_columns"]
            dx = self.params["node_spacing"]
        except KeyError:
            nr = 4
            nc = 5
            dx = 1.0

        # Create grid
        from landlab import RasterModelGrid

        self.grid = RasterModelGrid((nr, nc), dx)

        # Create and initialize elevation field
        # need to add starting elevation here and in hex grid. TODO

        self._create_synthetic_topography()
        # Set boundary conditions
        self._setup_synthetic_boundary_conditions()

    def _create_synthetic_topography(self):
        """Create topography for synthetic grids.

        If noise or initial elevation is added, it will only be added to the
        core nodes.
        """
        add_noise = self.params.get("add_random_noise", True)
        init_z = self.params.get("initial_elevation", 0.0)
        init_sigma = self.params.get("initial_noise_std", 0.0)
        seed = self.params.get("random_seed", 0)
        self.z = self.grid.add_zeros("node", "topographic__elevation")
        noise_location = self.params.get("add_noise_to_all_nodes", False)
        np.random.seed(seed)

        if noise_location:
            noise_nodes = np.arange(self.grid.size("node"))
        else:
            noise_nodes = self.grid.core_nodes

        if add_noise:
            rs = np.random.randn(noise_nodes.size)
            self.z[noise_nodes] += init_z + (init_sigma * rs)
        else:
            self.z[noise_nodes] += init_z

    def _setup_synthetic_boundary_conditions(self):
        """Set up boundary conditions for synthetic grids."""
        if self._starting_topography == "HexModelGrid":
            if "outlet_id" in self.params:
                self.opt_watershed = True
                self.outlet_node = self.params["outlet_id"]
            else:
                self.opt_watershed = False
                self.outlet_node = 0
                closed_boundaries = self.params.get("boundary_closed", False)
                if closed_boundaries:
                    self.grid.status_at_node[self.grid.boundary_nodes] = CLOSED_BOUNDARY

        else:
            if "outlet_id" in self.params:
                self.opt_watershed = True
                self.outlet_node = self.params["outlet_id"]
            else:
                self.opt_watershed = False
                self.outlet_node = 0
                east_closed = self.params.get("east_boundary_closed", False)
                north_closed = self.params.get("north_boundary_closed", False)
                west_closed = self.params.get("west_boundary_closed", False)
                south_closed = self.params.get("south_boundary_closed", False)

                self.grid.set_closed_boundaries_at_grid_edges(
                    east_closed, north_closed, west_closed, south_closed
                )

    def read_topography(self, file_path, name="topographic__elevation", halo=1):
        """Read and return topography from file.

        Parameters
        ----------
        file_path : str
            File path to read. Must be either a NetCDF or ESRI ASCII type file.
        name : str, optional
            Name of grid field for read topography. Default value is
             topographic__elevation.
        halo : int, optional
            Halo with which to pad DEM. Used only if file is an ESRI ASCII type.

        Returns
        -------
        (grid, vals) : tuple
          Model grid and value field.

        Examples
        --------
        We recommend that you look at the ``terrainbento`` tutorials for
        examples of usage.
        """
        try:
            (grid, vals) = read_esri_ascii(file_path, name=name, halo=halo)
        except:
            grid = read_netcdf(file_path)
            vals = grid.at_node[name]
        return (grid, vals)

    def get_parameter_from_exponent(self, param_name, raise_error=True):
        """Return absolute parameter value from provided exponent.

        Parameters
        ----------
        parameter_name : str
        raise_error : boolean
            Raise an error if parameter doesn not exist. Default is True.

        Returns
        -------
        value : float
          Parameter value.

        Examples
        --------
        >>> from landlab import HexModelGrid
        >>> from terrainbento import ErosionModel

        Sometimes in makes sense to provide a parameter as an exponent (base 10).
        If the string `'_exp'` is attached to the end of the name in the input
        dictionary, this function can help.

        >>> params = {'model_grid' : 'HexModelGrid',
        ...           'water_erodability_exp' : -3.,
        ...           'dt': 1, 'output_interval': 2., 'run_duration': 10.}
        >>> em = ErosionModel(params=params)
        >>> em.get_parameter_from_exponent('water_erodability')
        0.001

        Alternatively, the same call to the dictionary still works if the
        parameter was not provided as an exponent.

        >>> params = {'model_grid' : 'HexModelGrid',
        ...           'water_erodability' : 0.5,
        ...           'dt': 1, 'output_interval': 2., 'run_duration': 10.}
        >>> em = ErosionModel(params=params)
        >>> em.get_parameter_from_exponent('water_erodability')
        0.5

        """
        if (param_name in self.params) and (param_name + "_exp" in self.params):
            raise ValueError(
                "Parameter file includes both absolute value and"
                "exponent version of:" + param_name
            )

        if (param_name in self.params) and (param_name + "_exp" not in self.params):
            param = self.params[param_name]
        elif (param_name not in self.params) and (param_name + "_exp" in self.params):
            param = 10. ** float(self.params[param_name + "_exp"])
        else:
            if raise_error:
                raise ValueError(
                    "Parameter file includes neither absolute"
                    "value or exponent version of:" + param_name
                )
            else:
                param = None
        return param

    def calculate_cumulative_change(self):
        """Calculate cumulative node-by-node changes in elevation."""
        self.grid.at_node["cumulative_erosion__depth"] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["initial_topographic__elevation"]
        )

    def write_output(self, field_names=None):
        """Write output to file as a netCDF.

        Filenames will have the value of ``'output_filename'`` from the input
        file or paramter dictionary as the first part of the file name and the
        model run iteration as the second part of the filename.

        Parameters
        ----------
        output_fields : list of str, optional
            List of model grid fields to write as output. Default is to write
            out all fields.
        """
        self.calculate_cumulative_change()
        filename = self._out_file_name + str(self.iteration).zfill(4) + ".nc"
        try:
            write_raster_netcdf(
                filename, self.grid, names=field_names, format="NETCDF4"
            )
        except NotImplementedError:
            graph = Graph.from_dict(
                {
                    "y_of_node": self.grid.y_of_node,
                    "x_of_node": self.grid.x_of_node,
                    "nodes_at_link": self.grid.nodes_at_link,
                }
            )

            if field_names is None:
                field_names = self.grid.at_node.keys()

            for field_name in field_names:

                graph._ds.__setitem__(
                    field_name, ("node", self.grid.at_node[field_name])
                )

            graph.to_netcdf(path=filename, mode="w", format="NETCDF4")

        self.run_output_writers()

    def finalize__run_one_step(self, dt):
        # calculate model time
        self._model_time += dt

        # Update boundary conditions
        self.update_boundary_conditions(dt)

    def finalize(self):
        """Finalize model

        This base-class method does nothing. Derived classes can override
        it to run any required finalization steps.
        """
        pass

    def run_for(self, dt, runtime):
        """Run model without interruption for a specified time period.

        ``run_for`` runs the model for the duration ``runtime`` with model time
        steps of ``dt``.

        Parameters
        ----------
        dt : float
            Model run timestep.
        runtime : float
            Total duration for which to run model.
        """
        elapsed_time = 0.
        keep_running = True
        while keep_running:
            if elapsed_time + dt >= runtime:
                dt = runtime - elapsed_time
                keep_running = False
            self.run_one_step(dt)
            elapsed_time += dt

    def run(self, output_fields=None):
        """Run the model until complete.

        The model will run for the duration indicated by the input file or
        dictionary parameter ``'run_duration'``, at a time step specified by the
        parameter ``'dt'``, and create ouput at intervales of ``'output_duration'``.

        Parameters
        ----------
        output_fields : list of str, optional
            List of model grid fields to write as output. Default is to write
            out all fields.
        """
        if self.save_first_timestep:
            self.iteration = 0
            self.write_output(field_names=output_fields)
        total_run_duration = self.params["run_duration"]
        output_interval = self.params["output_interval"]
        self.iteration = 1
        time_now = self._model_time
        while time_now < total_run_duration:
            next_run_pause = min(time_now + output_interval, total_run_duration)
            self.run_for(self.params["dt"], next_run_pause - time_now)
            time_now = next_run_pause
            self.write_output(field_names=output_fields)
            self.iteration += 1

        # now that the model is finished running, execute finalize.
        self.finalize()

    def run_output_writers(self):
        """Run all output writers."""
        if self.output_writers is not None:
            for name in self.output_writers["class"]:
                self.output_writers[name].run_one_step()
            for function in self.output_writers["function"]:
                function(self)

    def update_boundary_conditions(self, dt):
        """Run all boundary handlers forward by dt.

        Parameters
        ----------
        dt : float
            Timestep in unit of model time.
        """
        # Run each of the baselevel handlers.
        if self.boundary_handler is not None:
            for handler_name in self.boundary_handler:
                self.boundary_handler[handler_name].run_one_step(dt)


def main():
    """Executes model."""
    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    erosion_model = ErosionModel(input_file=infile)
    erosion_model.run()


if __name__ == "__main__":
    main()
