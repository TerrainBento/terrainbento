# !/usr/env/python
"""Base class for common functions of all terrainbento erosion models."""

import os
import sys
import time as tm
import warnings

import numpy as np
import xarray as xr
import yaml
from landlab import ModelGrid, create_grid
from landlab.components import FlowAccumulator, NormalFault

from terrainbento.boundary_handlers import (
    CaptureNodeBaselevelHandler,
    GenericFuncBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
)
from terrainbento.clock import Clock
from terrainbento.output_writers import (
    GenericOutputWriter,
    OWSimpleNetCDF,
    StaticIntervalOutputClassAdapter,
    StaticIntervalOutputFunctionAdapter,
    StaticIntervalOutputWriter,
)
from terrainbento.precipitators import RandomPrecipitator, UniformPrecipitator
from terrainbento.runoff_generators import SimpleRunoff

_SUPPORTED_PRECIPITATORS = {
    "UniformPrecipitator": UniformPrecipitator,
    "RandomPrecipitator": RandomPrecipitator,
}
_SUPPORTED_RUNOFF_GENERATORS = {"SimpleRunoff": SimpleRunoff}

_VALID_PRECIPITATORS = (UniformPrecipitator, RandomPrecipitator)
_VALID_RUNOFF_GENERATORS = SimpleRunoff

_DEFAULT_PRECIPITATOR = {"UniformPrecipitator": {}}
_DEFAULT_RUNOFF_GENERATOR = {"SimpleRunoff": {}}

_SUPPORTED_BOUNDARY_HANDLERS = [
    "NormalFault",
    "PrecipChanger",
    "CaptureNodeBaselevelHandler",
    "NotCoreNodeBaselevelHandler",
    "SingleNodeBaselevelHandler",
    "GenericFuncBaselevelHandler",
]

_HANDLER_METHODS = {
    "NormalFault": NormalFault,
    "PrecipChanger": PrecipChanger,
    "CaptureNodeBaselevelHandler": CaptureNodeBaselevelHandler,
    "NotCoreNodeBaselevelHandler": NotCoreNodeBaselevelHandler,
    "SingleNodeBaselevelHandler": SingleNodeBaselevelHandler,
    "GenericFuncBaselevelHandler": GenericFuncBaselevelHandler,
}

_DEFAULT_OUTPUT_DIR = os.path.join(os.curdir, "output")


def _verify_boundary_handler(handler):
    bad_name = False
    bad_instance = False
    if isinstance(handler, str):
        if handler not in _SUPPORTED_BOUNDARY_HANDLERS:
            bad_name = True
    else:  # if a dictionary {name, handler}
        for key in handler:
            name = handler[key].__class__.__name__
            if name not in _SUPPORTED_BOUNDARY_HANDLERS:
                bad_instance = True

    if bad_name:
        raise ValueError(
            "Only supported boundary condition handlers are "
            "permitted. These include: {valid}".format(
                valid="\n".join(_SUPPORTED_BOUNDARY_HANDLERS)
            )
        )

    if bad_instance:
        raise ValueError(
            "An invalid instance of "
            + name
            + " was passed as a boundary handler."
            + str(handler)
        )


def _setup_precipitator_or_runoff(grid, params, supported):
    """"""
    if len(params) != 1:
        raise ValueError(
            "Too many values provided to set up precipitator or runoff_generator"
        )
    for name in params:
        constructor = supported[name]
        instance = constructor(grid, **params[name])
    return instance


def _setup_boundary_handlers(grid, name, params):
    """Setup BoundaryHandlers for use by a terrainbento model.

    A boundary condition handler is a class with a **run_one_step** method
    that takes the parameter ``step``. Permitted boundary condition handlers
    include the Landlab Component **NormalFault** as well as the following
    options from terrainbento: **PrecipChanger**,
    **CaptureNodeBaselevelHandler**, **NotCoreNodeBaselevelHandler**,
    **SingleNodeBaselevelHandler**.

    Parameters
    ----------
    handler : str
        Name of a supported boundary condition handler.
    """
    _verify_boundary_handler(name)
    # Instantiate handler
    handler_func = _HANDLER_METHODS[name]
    boundary_handler = handler_func(grid, **params)

    return boundary_handler


class ErosionModel:
    """Base class providing common functionality for terrainbento models.

    An **ErosionModel** is the skeleton for the models of terrain evolution in
    terrainbento.

    This is a base class that does not implement any processes, but rather
    simply handles I/O and setup. Derived classes are meant to include
    Landlab components to model actual erosion processes.

    It is expected that a derived model will define an **__init__** and a
    **run_one_step** method. If desired, the derived model can overwrite the
    existing **run_for**, **run**, and **finalize** methods.

    The following at-node fields must be specified in the grid:
        - ``topographic__elevation``
    """

    _required_fields = ["topographic__elevation"]

    # Setup
    @classmethod
    def from_file(cls, file_like):
        """Construct a terrainbento model from a file.

        Parameters
        ----------
        file_like : file_like or str
            Contents of a parameter file, a file-like object, or the path to
            a parameter file.

        Examples
        --------
        >>> from io import StringIO
        >>> filelike = StringIO(
        ...     '''
        ... grid:
        ...   RasterModelGrid:
        ...     - [4, 5]
        ...     - fields:
        ...         node:
        ...           topographic__elevation:
        ...             constant:
        ...               - value: 0.0
        ... clock:
        ...   step: 1
        ...   stop: 200
        ... '''
        ... )
        >>> model = ErosionModel.from_file(filelike)
        >>> model.clock.step
        1.0
        >>> model.clock.stop
        200.0
        >>> model.grid.shape
        (4, 5)
        """
        # first get contents.
        try:
            contents = file_like.read()
        except AttributeError:  # was a str
            if os.path.isfile(file_like):
                with open(file_like) as fp:
                    contents = fp.read()
            else:
                contents = file_like  # not tested

        # then parse contents.
        params = yaml.safe_load(contents)

        # construct instance
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params, output_writers=None):
        """Construct a terrainbento model from an input parameter dictionary.

        The input parameter dictionary portion associated with the "grid"
        keword will be passed directly to the Landlab
        `create_grid <https://landlab.readthedocs.io/en/master/reference/grid/create.html#landlab.grid.create.create_grid>`_.
        function.

        Parameters
        ----------
        params : dict
            Dictionary of input parameters.
        output_writers : dictionary of output writers.
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot). There are two formats for the dictionary entries:

            1) Items can have a key of "class" or "function" and a value of
               a list of simple output classes (uninstantiated) or
               functions, respectively. All output writers defined this way
               will use the `output_interval` provided to the ErosionModel
               constructor.
            2) Items can have a key with any unique string representing the
               output writer's name and a value containing a dict with the
               uninstantiated class and arguments. The value follows the
               format:

               .. code-block:: python

                   {
                       "class": MyWriter,
                       "args": [],  # optional
                       "kwargs": {},  # optional
                   }

               where `args` and `kwargs` are passed to the constructor for
               `MyWriter`. `MyWriter` must be a child class of
               GenericOutputWriter.

               The two formats can be present simultaneously. See the Jupyter
               notebook examples for more details.

        Examples
        --------
        >>> params = {
        ...     "grid": {
        ...         "RasterModelGrid": [
        ...             (4, 5),
        ...             {
        ...                 "fields": {
        ...                     "node": {
        ...                         "topographic__elevation": {
        ...                             "constant": [{"value": 0.0}]
        ...                         }
        ...                     }
        ...                 }
        ...             },
        ...         ]
        ...     },
        ...     "clock": {"step": 1, "stop": 200},
        ... }
        >>> model = ErosionModel.from_dict(params)
        >>> model.clock.step
        1.0
        >>> model.clock.stop
        200.0
        >>> model.grid.shape
        (4, 5)
        """
        cls._validate(params)

        # grid, clock
        grid = create_grid(params.pop("grid"))
        clock = Clock.from_dict(params.pop("clock"))

        # precipitator
        precip_params = params.pop("precipitator", _DEFAULT_PRECIPITATOR)
        precipitator = _setup_precipitator_or_runoff(
            grid, precip_params, _SUPPORTED_PRECIPITATORS
        )

        # runoff_generator
        runoff_params = params.pop("runoff_generator", _DEFAULT_RUNOFF_GENERATOR)
        runoff_generator = _setup_precipitator_or_runoff(
            grid, runoff_params, _SUPPORTED_RUNOFF_GENERATORS
        )

        # boundary_handlers
        boundary_handlers = params.pop("boundary_handlers", {})
        bh_dict = {}
        for name in boundary_handlers:
            bh_params = boundary_handlers[name]
            bh_dict[name] = _setup_boundary_handlers(grid, name, bh_params)

        # create instance
        return cls(
            clock,
            grid,
            precipitator=precipitator,
            runoff_generator=runoff_generator,
            boundary_handlers=bh_dict,
            output_writers=output_writers,
            **params,
        )

    @classmethod
    def _validate(cls, params):
        """Make sure necessary things for a model grid and a clock are here."""
        if "grid" not in params:
            raise ValueError("No grid provided as part of input parameters")
        if "clock" not in params:
            raise ValueError("No clock provided as part of input parameters")

    def __init__(
        self,
        clock,
        grid,
        precipitator=None,
        runoff_generator=None,
        flow_director="FlowDirectorSteepest",
        depression_finder=None,
        flow_accumulator_kwargs=None,
        boundary_handlers=None,
        output_writers=None,
        output_default_netcdf=True,
        output_interval=None,
        save_first_timestep=True,
        save_last_timestep=True,
        output_prefix="terrainbento-output",
        output_dir=_DEFAULT_OUTPUT_DIR,
        fields=None,
    ):
        """
        Parameters
        ----------
        clock : terrainbento Clock instance
        grid : landlab model grid instance
            The grid must have all required fields.
        precipitator : terrainbento precipitator, optional
            An instantiated version of a valid precipitator. See the
            :py:mod:`precipitator <terrainbento.precipitator>` module for
            valid options. The precipitator creates rain. Default value is the
            :py:class:`UniformPrecipitator` with a rainfall flux of 1.0.
        runoff_generator : terrainbento runoff_generator, optional
            An instantiated version of a valid runoff generator. See the
            :py:mod:`runoff generator <terrainbento.runoff_generator>` module
            for valid options. The runoff generator converts rain into runoff.
            This runoff is then accumulated into surface water discharge
            (:math:`Q`) and used by channel erosion components. Default value
            is :py:class:`SimpleRunoff` in which all rainfall turns into
            runoff. For the drainage area version of the stream power law use
            the default precipitator and runoff_generator.

            If the default values of both the precipitator and
            runoff_generator are used, then :math:`Q` will be equal to drainage
            area.

        flow_director : str, optional
            String name of a
            `Landlab FlowDirector <https://landlab.readthedocs.io/en/master/reference/components/flow_director.html>`_.
            Default is "FlowDirectorSteepest".
        depression_finder : str, optional
            String name of a Landlab depression finder. Default is None.
        flow_accumulator_kwargs : dictionary, optional
            Dictionary of any additional keyword arguments to pass to the
            `Landlab FlowAccumulator <https://landlab.readthedocs.io/en/master/reference/components/flow_accum.html>`_.
            Default is an empty dictionary.
        boundary_handlers : dictionary, optional
            Dictionary with ``name: instance`` key-value pairs. Each entry
            must be a valid instance of a terrainbento boundary handler. See
            the :py:mod:`boundary handlers <terrainbento.boundary_handlers>`
            module for valid options.
        output_writers : dictionary of output writers.
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot). There are two formats for the dictionary entries:

            1. ("Old style") Items can have a key of "class" or "function"
               and a value that is a list of uninstantiated output classes
               or a list of output functions, respectively. All output
               writers defined this way will use the **output_interval**
               argument provided to the ErosionModel constructor.
            2. ("New style") Items can have a key of any unique string
               representing the output writer's name and a value that is a
               dictionary containing the uninstantiated class and any
               arguments. The dictionary follows the format:

               .. code-block:: python

                   {
                       "class": MyWriter,
                       "args": [],  # optional
                       "kwargs": {},  # optional
                   }

               where `args` and `kwargs` are passed to the constructor for
               `MyWriter`. All new style output writers must be a child
               class of GenericOutputWriter. The ErosionModel reference is
               automatically prepended to args.

            The two formats can be present simultaneously. See the Jupyter
            notebook examples for more details.
        output_default_netcdf : bool, optional
            Indicates whether the erosion model should automatically create a
            simple netcdf output writer which behaves identical to the built-in
            netcdf writer from older terrainbento versions. Uses the
            'output_interval' argument as the output interval. Defaults to True.
        output_interval : float, optional
            The time between output calls for old-style output writers and the
            default netcdf writer. Default is the Clock's stop time.
        save_first_timestep : bool, optional
            Indicates whether model output should be saved at time zero (the
            initial conditions). This affects old and new style output writers.
            Default is True.
        save_last_timestep : bool, optional
            Indicates that the last output time must be at the clock stop time.
            This affects old and new style output writers. Defaults to True.
        output_prefix : str, optional
            String prefix for names of all output files. Default is
            ``"terrainbento-output"``.
        output_dir : string, optional
            Directory that output should be saved to. Defaults to an "output"
            directory in the current directory.
        fields : list, optional
            List of field names to write as netCDF output. Default is to only
            write out "topographic__elevation".

        Returns
        -------
        ErosionModel: object

        Examples
        --------
        This model is a base class and is not designed to be run on its own. We
        recommend that you look at the terrainbento tutorials for examples of
        usage.
        """
        flow_accumulator_kwargs = flow_accumulator_kwargs or {}
        boundary_handlers = boundary_handlers or {}
        output_writers = output_writers or {}
        fields = fields or ["topographic__elevation"]
        # type checking
        if isinstance(clock, Clock) is False:
            raise ValueError("Provided Clock is not valid.")
        if isinstance(grid, ModelGrid) is False:
            raise ValueError("Provided Grid is not valid.")

        # save the grid, clock, and parameters.
        self.grid = grid
        self.clock = clock

        # first pass of verifying fields
        self._verify_fields(self._required_fields)

        # save reference to elevation
        self.z = grid.at_node["topographic__elevation"]

        self.grid.add_zeros("node", "cumulative_elevation_change")

        self.grid.add_field("node", "initial_topographic__elevation", self.z.copy())

        # save output_information
        self.save_first_timestep = save_first_timestep
        self.save_last_timestep = save_last_timestep
        self._output_prefix = output_prefix
        self.output_dir = output_dir
        self.output_fields = fields
        self._output_files = []
        if output_interval is None:
            output_interval = clock.stop
        self.output_interval = output_interval

        # instantiate model time.
        self._model_time = 0.0

        # instantiate container for computational timestep:
        self._compute_time = [tm.time()]

        ###################################################################
        # address Precipitator and RUNOFF_GENERATOR
        ###################################################################

        # verify that precipitator is valid
        if precipitator is None:
            precipitator = UniformPrecipitator(self.grid)
        else:
            if isinstance(precipitator, _VALID_PRECIPITATORS) is False:
                raise ValueError("Provided value for precipitator not valid.")
        self.precipitator = precipitator

        # verify that runoff_generator is valid
        if runoff_generator is None:
            runoff_generator = SimpleRunoff(self.grid)
        else:
            if isinstance(runoff_generator, _VALID_RUNOFF_GENERATORS) is False:
                raise ValueError("Provide value for runoff_generator not valid.")
        self.runoff_generator = runoff_generator

        ###################################################################
        # instantiate flow direction and accumulation
        ###################################################################
        # Instantiate a FlowAccumulator, if DepressionFinder is provided
        # AND director = Steepest, then we need routing to be D4,
        # otherwise, just passing params should be sufficient.
        if (depression_finder is not None) and (
            flow_director == "FlowDirectorSteepest"
        ):
            self.flow_accumulator = FlowAccumulator(
                self.grid,
                routing="D4",
                depression_finder=depression_finder,
                **flow_accumulator_kwargs,
            )
        else:
            self.flow_accumulator = FlowAccumulator(
                self.grid,
                flow_director=flow_director,
                depression_finder=depression_finder,
                **flow_accumulator_kwargs,
            )

        if self.flow_accumulator.depression_finder is None:
            self._erode_flooded_nodes = True
        else:
            self._erode_flooded_nodes = False

        ###################################################################
        # Boundary Conditions and Output Writers
        ###################################################################
        _verify_boundary_handler(boundary_handlers)
        self.boundary_handlers = boundary_handlers

        # Instantiate all the output writers and store in a list
        self.all_output_writers = self._setup_output_writers(
            output_writers,
            output_default_netcdf,
        )

        # Keep track of when each writer needs to write next
        self.active_output_times = {}  # {next time : [writers]}
        self.sorted_output_times = []  # sorted list of the next output times
        for ow_writer in self.all_output_writers:
            first_time = ow_writer.advance_iter()
            self._update_output_times(ow_writer, first_time, None)

    def _verify_fields(self, required_fields):
        """Verify all required fields are present."""
        for field in required_fields:
            if field not in self.grid.at_node:
                raise ValueError(f"Required field {field} not present.")

    def _setup_output_writers(self, output_writers, output_default_netcdf):
        """Convert all output writers to the new style and instantiate output
        writer classes.

        Parameters
        ----------
        output_writers : dictionary of output writers.
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot). There are two formats for the dictionary entries:

            1) ("Old style") Items can have a key of "class" or "function"
               and a value that is a list of uninstantiated output classes
               or a list of output functions, respectively. All output
               writers defined this way will use the **output_interval**
               argument provided to the ErosionModel constructor.
            2) ("New style") Items can have a key of any unique string
               representing the output writer's name and a value that is a
               dictionary containing the uninstantiated class and any
               arguments. The dictionary follows the format: {
                    'class' : MyWriter,
                    'args' : [], # optional
                    'kwargs' : {}, # optional
                    }
               where `args` and `kwargs` are passed to the constructor for
               `MyWriter`. All new style output writers must be a child
               class of GenericOutputWriter. The ErosionModel reference is
               automatically prepended to args.

            The two formats can be present simultaneously. See the Jupyter
            notebook examples for more details.

        output_default_netcdf : bool, optional
            Indicates whether the erosion model should automatically create a
            simple netcdf output writer which behaves identical to the built-in
            netcdf writer from older terrainbento versions. Uses the
            'output_interval' argument as the output interval. Defaults to True.

        Returns
        -------
        instantiated_output_writers : list of GenericOutputWriter objects
            A list of instantiated output writers all based on the
            GenericOutputWriter.

        Notes
        -----
        All classes and functions provided in the 'class' and 'function'
        entries in the output_writer dictionary will be given to an adapter
        class for StaticIntervalOutputWriter so that they can be used with the
        new framework. All of theses writers will use `output_interval`.

        """

        # Note: I can't guarantee that the names will stay unique. I need to
        # convert the 'class' and 'function' writers to the new style and there
        # is a non-zero chance the user happens to use a name for the new style
        # that has the same name that I give the converted writers. These names
        # are used for output filenames, so I don't want to use anything ugly.
        # Hence why I return a list instead of another dictionary.

        # Add a default netcdf writer if desired.
        assert isinstance(output_default_netcdf, bool)
        if output_default_netcdf:
            output_writers["simple-netcdf"] = {
                "class": OWSimpleNetCDF,
                "args": [self.output_fields],
                "kwargs": {
                    "intervals": self.output_interval,
                    "add_id": True,
                    "output_dir": self.output_dir,
                },
            }

        instantiated_writers = []
        for name in output_writers:
            if name == "class":
                # Old style class output writers. Give information to an
                # adapter for instantiating as a static interval writer.
                for ow_class in output_writers["class"]:
                    new_writer = StaticIntervalOutputClassAdapter(
                        model=self,
                        output_interval=self.output_interval,
                        ow_class=ow_class,
                        save_first_timestep=self.save_first_timestep,
                        save_last_timestep=self.save_last_timestep,
                        output_dir=self.output_dir,
                    )
                    # new_name = new_writer.name
                    # assert new_name not in instantiated_writers, \
                    #        f"Output writer '{name}' already exists"
                    instantiated_writers.append(new_writer)

            elif name == "function":
                # Old style function output writers. Give information to an
                # adapter for instantiating as a static interval writer.
                for ow_function in output_writers["function"]:
                    new_writer = StaticIntervalOutputFunctionAdapter(
                        model=self,
                        output_interval=self.output_interval,
                        ow_function=ow_function,
                        save_first_timestep=self.save_first_timestep,
                        save_last_timestep=self.save_last_timestep,
                        output_dir=self.output_dir,
                    )
                    instantiated_writers.append(new_writer)

            else:
                # New style output writer class
                writer_dict = output_writers[name]
                assert isinstance(
                    writer_dict, dict
                ), "The new style output writer entry must be a dictionary"
                assert "class" in writer_dict, "".join(
                    [
                        f"New style output writer {name} must have a 'class'",
                        "entry",
                    ]
                )
                ow_class = writer_dict["class"]
                ow_args = writer_dict.get("args", [self])
                ow_kwargs = writer_dict.get("kwargs", {})

                # Prepend a reference to the model to the args (if not there)
                if ow_args and ow_args[0] is not self:
                    ow_args = [self] + ow_args
                elif ow_args is None:  # pragma: no cover
                    ow_args = [self]

                # Add some kwargs if they were not already provided
                defaults = {
                    "name": name,
                    "save_first_timestep": self.save_first_timestep,
                    "save_last_timestep": self.save_last_timestep,
                    "output_dir": self.output_dir,
                }
                if issubclass(ow_class, StaticIntervalOutputWriter):
                    if "times" not in ow_kwargs:
                        # Using a static interval writer and no times provided,
                        # use the output_interval as a default interval.
                        defaults["intervals"] = self.output_interval
                defaults.update(ow_kwargs)
                ow_kwargs = defaults

                new_writer = ow_class(*ow_args, **ow_kwargs)
                instantiated_writers.append(new_writer)

        return instantiated_writers

    # Attributes
    @property
    def model_time(self):
        """Return current time of model integration in model time units."""
        return self._model_time

    @property
    def next_output_time(self):
        """Return the next output time in model time units. If there are no
        more active output writers, return np.inf instead."""
        if self.sorted_output_times:

            return self.sorted_output_times[0]
        else:
            return np.inf

    @property
    def output_prefix(self):
        """Model prefix for output filenames."""
        return self._output_prefix

    @property
    def _out_file_name(self):
        """(Deprecated) Get the filename model prefix. Used to get the netcdf
        filename base."""
        warnings.warn(
            " ".join(
                [
                    "ErosionModel's _out_file_name is no longer available.",
                    "Getting _output_prefix instead, but may not behave as expected.",
                    "Please use the 'output_prefix' argument in the constructor.",
                ]
            ),
            DeprecationWarning,
        )
        return self._output_prefix

    @_out_file_name.setter
    def _out_file_name(self, prefix):
        """(Deprecated) Set the filename model prefix. Used to set the netcdf
        filename base."""
        warnings.warn(
            " ".join(
                [
                    "ErosionModel's _out_file_name is no longer available.",
                    "Setting _output_prefix instead, but may not behave as expected.",
                    "Please use the 'output_prefix' argument in the constructor.",
                ]
            ),
            DeprecationWarning,
        )
        self._output_prefix = prefix

    # Model run methods
    def calculate_cumulative_change(self):
        """Calculate cumulative node-by-node changes in elevation."""
        self.grid.at_node["cumulative_elevation_change"][:] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["initial_topographic__elevation"]
        )

    def create_and_move_water(self, step):
        """Create and move water.

        Run the precipitator, the runoff generator, and the flow
        accumulator, in that order.
        """
        self.precipitator.run_one_step(step)
        self.runoff_generator.run_one_step(step)
        self.flow_accumulator.run_one_step()

    def finalize__run_one_step(self, step):
        """Finalize run_one_step method.

        This base-class method increments model time and updates
        boundary conditions.
        """
        # calculate model time
        self._model_time += step

        # Update boundary conditions
        self.update_boundary_conditions(step)

    def finalize(self):
        """Finalize model.

        This base-class method does nothing. Derived classes can
        override it to run any required finalization steps.
        """
        pass

    def run_for(self, step, runtime):
        """Run model without interruption for a specified time period.

        ``run_for`` runs the model for the duration ``runtime`` with model time
        steps of ``step``.

        Parameters
        ----------
        step : float
            Model run timestep.
        runtime : float
            Total duration for which to run model.
        """
        elapsed_time = 0.0
        keep_running = True
        while keep_running:
            if elapsed_time + step >= runtime:
                step = runtime - elapsed_time
                keep_running = False
            self.run_one_step(step)
            elapsed_time += step

    def run(self):
        """Run the model until complete.

        The model will run for the duration indicated by the input file
        or dictionary parameter ``"stop"``, at a time step specified by
        the parameter ``"step"``, and create ouput at intervals specified by
        the individual output writers.
        """
        self._itters = []

        if self.save_first_timestep:
            self.iteration = 0
            self._itters.append(0)
            self.calculate_cumulative_change()
            self.write_output()
        self.iteration = 1
        time_now = self._model_time
        while time_now < self.clock.stop:
            next_run_pause = min(
                # time_now + self.output_interval, self.clock.stop,
                self.next_output_time,
                self.clock.stop,
            )
            assert next_run_pause > time_now
            self.run_for(self.clock.step, next_run_pause - time_now)
            time_now = self._model_time
            self._itters.append(self.iteration)
            self.calculate_cumulative_change()
            self.write_output()
            self.iteration += 1

        # now that the model is finished running, execute finalize.
        self.finalize()

    def _ensure_precip_runoff_are_vanilla(self, vsa_precip=False):
        """Ensure only default versions of precipitator/runoff are used.

        Some models only work when the precipitator and runoff generator
        are the default versions.
        """
        if isinstance(self.precipitator, UniformPrecipitator) is False:
            raise ValueError("This model must be run with a UniformPrecipitator.")

        if vsa_precip is False:
            if self.precipitator._rainfall_flux != 1:
                raise ValueError("This model must use a rainfall__flux value of 1.0.")

        # if isinstance(self.runoff_generator, SimpleRunoff) is False:
        #     raise ValueError("This model must be run with SimpleRunoff.")

        if self.runoff_generator.runoff_proportion != 1.0:
            raise ValueError("The model must use a runoff_proportion of 1.0.")

    def update_boundary_conditions(self, step):
        """Run all boundary handlers forward by step.

        Parameters
        ----------
        step : float
            Timestep in unit of model time.
        """
        # Run each of the baselevel handlers.
        for name in self.boundary_handlers:
            self.boundary_handlers[name].run_one_step(step)

    # Output methods
    def write_output(self):
        """Run output writers if it is the correct model time."""

        # assert that the model has not passed the next output time.
        assert self._model_time <= self.next_output_time, "".join(
            [
                f"Model time (t={self._model_time}) has passed the next ",
                f"output time (t={self.next_output_time})",
            ]
        )

        if self._model_time == self.next_output_time:
            # The current model time matches the next output time
            current_time = self.sorted_output_times.pop(0)
            current_writers = self.active_output_times.pop(current_time)
            for ow_writer in current_writers:
                # Run all the output writers associated with this time.
                ow_writer.run_one_step()
                next_time = ow_writer.advance_iter()
                self._update_output_times(ow_writer, next_time, current_time)

    def _update_output_times(self, ow_writer, new_time, current_time):
        """Private method to update the dictionary of active output writers
        and the sorted list of next output times.

        Parameters
        ----------
        ow_writer : GenericOutputWriter object
            The output writer that has just finished writing output and advanced
            it's time iterator.
        new_time : float
            The next time that the output writer will need to write output.
        current_time : float
            The current model time.

        Notes
        -----
        This function enforces that output times align with model steps. If an
        output writer returns a next_time that is in between model steps, then
        the output time is delayed to the following step and a warning is
        generated. This function may generate skip warnings if the subsequent
        next times are less than the delayed step time.

        """
        if new_time is None:
            # The output writer has exhausted all of it's output times.
            # Do not add it back to the active dict/list
            return

        model_step = self.clock.step
        if current_time is not None:
            try:
                assert new_time > current_time
            except AssertionError:
                warnings.warn(
                    "".join(
                        [
                            f"The output writer {ow_writer.name} is providing a ",
                            "next time that is less than or equal to the current ",
                            "time. Possibly because the previous time was in ",
                            "between steps, delaying the output until now. ",
                            "Skipping ahead.",
                        ]
                    )
                )
                for n_skips in range(10):
                    # Allow 10 attempts to skip
                    new_time = ow_writer.advance_iter()
                    if new_time is None:
                        # iterator exhausted. No more processing needed
                        return
                    elif new_time > current_time:
                        break
                else:
                    # Could not find a suitable next_time
                    raise AssertionError(
                        "".join(
                            [
                                "Output writer failed to return a next time greater ",
                                "than the current time after several attempts.",
                            ]
                        )
                    )

        # See if the new output time aligns with the model step.
        if (new_time % model_step) != 0.0:
            warnings.warn(
                "".join(
                    [
                        f"Output writer {ow_writer.name} is requesting a ",
                        "time that is not divisible by the model step. ",
                        "Delaying output to the following step.\n",
                        f"Output time = {new_time}\n",
                        f"Model step = {model_step}\n",
                        f"Remainder = {new_time % model_step}\n\n",
                    ]
                )
            )
            new_time = np.ceil(new_time / model_step) * model_step

        # Add the writer to the active_output_times dict
        if new_time in self.active_output_times:
            # New time is already in the active_output_times dictionary
            self.active_output_times[new_time].append(ow_writer)
        else:
            # New time is not in the active_output_times dictionary
            # Add it to the dict and resort the output times list
            self.active_output_times[new_time] = [ow_writer]
            self.sorted_output_times = sorted(self.active_output_times)

    def to_xarray_dataset(
        self,
        time_unit="time units",
        reference_time="model start",
        space_unit="space units",
    ):
        """Convert model output to an xarray dataset.

        If you would like to have CF compliant NetCDF make sure that your time
        and space units and reference times will work with standard decoding.

        The default time unit and reference time will give the time dimention a
        value of "time units since model start". The default space unit will
        give a value of "space unit".

        Parameters
        ----------
        time_unit: str, optional
            Name of time unit. Default is "time units".
        reference time: str, optional
            Reference tim. Default is "model start".
        space_unit: str, optional
            Name of space unit. Default is "space unit".
        """
        # open all files as a xarray dataset
        ds = xr.open_mfdataset(
            self.get_output(extension="nc"),
            concat_dim="nt",
            engine="netcdf4",
            combine="nested",
            data_vars=self.output_fields,
        )

        # add a time dimension
        time_array = np.asarray(self._itters) * self.output_interval
        time = xr.DataArray(
            time_array,
            dims=("nt"),
            attrs={
                "units": time_unit + " since " + reference_time,
                "standard_name": "time",
            },
        )

        ds["time"] = time

        # set x and y to coordinates
        ds = ds.set_coords(["x", "y", "time"])

        # rename dimensions
        ds = ds.rename(name_dict={"ni": "x", "nj": "y", "nt": "time"})

        # set x and y units
        ds["x"] = xr.DataArray(ds.x, dims=("x"), attrs={"units": space_unit})
        ds["y"] = xr.DataArray(ds.y, dims=("y"), attrs={"units": space_unit})

        return ds

    def save_to_xarray_dataset(
        self,
        filename="terrainbento.nc",
        time_unit="time units",
        reference_time="model start",
        space_unit="space units",
    ):
        """Save model output to xarray dataset.

        If you would like to have CF compliant NetCDF make sure that your time
        and space units and reference times will work with standard decoding.

        The default time unit and reference time will give the time dimention a
        value of "time units since model start". The default space unit will
        give a value of "space unit".

        Parameters
        ----------
        filename: str, optional
            The file path where the file should be saved. The default value is
            "terrainbento.nc".
        time_unit: str, optional
            Name of time unit. Default is "time units".
        reference time: str, optional
            Reference tim. Default is "model start".
        space_unit: str, optional
            Name of space unit. Default is "space unit".
        """
        ds = self.to_xarray_dataset(
            time_unit=time_unit,
            space_unit=space_unit,
            reference_time=reference_time,
        )
        ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4")
        ds.close()

    def _format_extension_and_writer_args(self, extension, writer):
        """Private method to parse the extension and writer arguments for the
        **remove_output** and **get_output** functions.

        Parameters
        ----------
        extension : string or list of strings or None
            Specify the type(s) of files to look for.
        writer : GenericOutputWriter instance or list of instances or string or list of strings or None
            Specify which output writers to look at either by the writer's
            handle or by the writer's name.

        Returns
        -------
        extension_list : list of strings
            List of strings representing the types of files to look for. Can
            return [None] indicating all files.
        writer_list : list of GenericOutputWriters
            List of GenericOutputWriter instances to look at. Can
            return [None] indicating all output writers should be looked at.
        """

        extension_list = None
        writer_list = None

        if isinstance(writer, GenericOutputWriter):
            # Writer argument is an object, convert to a list
            writer_list = [writer]
        elif isinstance(writer, str):
            # Writer argument is the name of the writer, get object
            writer_list = self.get_output_writer(writer)
        elif isinstance(writer, list):
            # Writer argument is a list
            writer_list = []
            # Check what is in the list
            for i, w in enumerate(writer):
                if isinstance(w, GenericOutputWriter):
                    writer_list.append(w)
                elif isinstance(w, str):
                    # Item is a name, replace with the object
                    found_writers = self.get_output_writer(w)
                    writer_list += found_writers
                else:  # pragma: no cover
                    raise TypeError(f"Unrecognized writer argument. {w}")
        elif writer is None:
            # Default to all writers
            writer_list = self.all_output_writers
        else:  # pragma: no cover
            raise TypeError(f"Unrecognized writer argument. {writer}")

        if isinstance(extension, str):
            # Extension argument is a string
            extension_list = [extension]
        elif isinstance(extension, list):
            # Extension argument is a list of strings
            extension_list = extension
            assert all([isinstance(e, str) for e in extension_list])
        elif extension is None:
            # Default to all extensions
            extension_list = [None]
        else:  # pragma: no cover
            raise TypeError(f"Unrecognized extension argument. {extension}")

        return extension_list, writer_list

    def remove_output_netcdfs(self):
        """Remove netcdf output files written during a model run. Only works
        for new style writers including the default netcdf writer."""
        self.remove_output(extension="nc")

    def remove_output(self, extension=None, writer=None):
        """Remove files written by new style writers during a model
        run. Does not work for old style writers which have no way to report
        what they have written. Can specify types of files and/or writers.

        To do: allow 'writer' to be a string for the name of the writer?

        Parameters
        ----------
        extension : string or list of strings, optional
            Specify what type(s) of files should be deleted. Defaults to None
            which deletes all file types. Don't include a leading period.
        writer : GenericOutputWriter instance or list of instances or string or list of strings or None
            Specify if the files should come from certain output writers either
            by the writer's handle or by the writer's name. Defaults to
            deleting files from all writers.

        """

        lists = self._format_extension_and_writer_args(extension, writer)
        extension_list, writer_list = lists

        for ow in writer_list:
            assert ow is not None
            for ext in extension_list:
                assert ext is None or isinstance(ext, str)
                if ext and ext[0] == ".":
                    ext = ext[1:]  # ignore leading period if present
                ow.delete_output_files(ext)

    def get_output(self, extension=None, writer=None):
        """Get a list of filepaths for files written by new style writers
        during a model run. Does not work for old style writers which have no
        way to report what they have written.  Can specify types of files
        and/or writers.

        Parameters
        ----------
        extension : string or list of strings, optional
            Specify what type(s) of files should be returned. Defaults to None
            which returns all file types. Don't include a leading period.
        writer : GenericOutputWriter instance or list of instances or string or list of strings or None
            Specify if the files should come from certain output writers either
            by the writer's handle or by the writer's name. Defaults to
            returning files from all writers.

        Returns
        -------
        filepaths : list of strings
            A list of filepath strings that match the desired extensions and
            writers.
        """

        lists = self._format_extension_and_writer_args(extension, writer)
        extension_list, writer_list = lists

        output_list = []
        for ow in writer_list:
            assert ow is not None
            for ext in extension_list:
                assert ext is None or isinstance(ext, str)
                output_list += ow.get_output_filepaths(ext)

        return output_list

    def get_output_writer(self, name):
        """Get the references for object writer(s) from the writer's name.

        Parameters
        ----------
        name : string
            The name of the output writer to look for. Can match multiple
            writers.

        Returns
        -------
        matches : list of GenericOutputWriter objects
            The list of any GenericOutputWriter whose name contains the
            argument name string. Will return an empty list if there are no
            matches.

        """
        matches = []
        for ow in self.all_output_writers:
            if name in ow.name:
                matches.append(ow)
        return matches


def main():  # pragma: no cover
    """Executes model."""
    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    erosion_model = ErosionModel.from_file(infile)
    erosion_model.run()


if __name__ == "__main__":  # pragma: no cover
    main()
