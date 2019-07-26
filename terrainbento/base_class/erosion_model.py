# coding: utf8
# !/usr/env/python
"""Base class for common functions of all terrainbento erosion models."""

import os
import sys
import time as tm

import dask
import numpy as np
import xarray as xr
import yaml

from landlab import ModelGrid, create_grid
from landlab.components import FlowAccumulator, NormalFault
from landlab.graph import Graph
from landlab.io.netcdf import write_raster_netcdf
from terrainbento.boundary_handlers import (
    CaptureNodeBaselevelHandler,
    GenericFuncBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
)
from terrainbento.clock import Clock
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
            (
                "Only supported boundary condition handlers are "
                "permitted. These include: {valid}".format(
                    valid="\n".join(_SUPPORTED_BOUNDARY_HANDLERS)
                )
            )
        )

    if bad_instance:
        raise ValueError(
            (
                "An invalid instance of "
                + name
                + " was passed as a boundary handler."
                + str(handler)
            )
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


class ErosionModel(object):

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
        >>> filelike = StringIO('''
        ... grid:
        ...   RasterModelGrid:
        ...     - [4, 5]
        ...     - fields:
        ...         node:
        ...           topographic__elevation:
        ...             constant:
        ...               - value: 0
        ... clock:
        ...   step: 1
        ...   stop: 200
        ... ''')
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
                with open(file_like, "r") as fp:
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
        `create_grid <https://landlab.readthedocs.io/en/latest/landlab.grid.create.html>`_.
        function.

        Parameters
        ----------
        params : dict
            Dictionary of input parameters.
        output_writers : dictionary of output writers.
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot). These should be passed in a dictionary with two
            keys: "class" and "function". The value associated with each of
            these should be a list containing the uninstantiated output
            writers. See the Jupyter notebook examples for more details.

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
        ...                             "constant": [{"value": 0}]
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
        runoff_params = params.pop(
            "runoff_generator", _DEFAULT_RUNOFF_GENERATOR
        )
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
            **params
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
        output_interval=None,
        save_first_timestep=True,
        output_prefix="terrainbento_output",
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
            `Landlab FlowDirector <https://landlab.readthedocs.io/en/latest/landlab.components.flow_director.html>`_.
            Default is "FlowDirectorSteepest".
        depression_finder : str, optional
            String name of a Landlab depression finder. Default is None.
        flow_accumulator_kwargs : dictionary, optional
            Dictionary of any additional keyword arguments to pass to the
            `Landlab FlowAccumulator <https://landlab.readthedocs.io/en/latest/landlab.components.flow_accum.html>`_.
            Default is an empty dictionary.
        boundary_handlers : dictionary, optional
            Dictionary with ``name: instance`` key-value pairs. Each entry
            must be a valid instance of a terrainbento boundary handler. See
            the :py:mod:`boundary handlers <terrainbento.boundary_handlers>`
            module for valid options.
        output_writers : dictionary of output writers.
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot). These should be passed in a dictionary with two
            keys: "class" and "function". The value associated with each of
            these should be a list containing the uninstantiated output
            writers. See the Jupyter notebook examples for more details.
        output_interval : float, optional
            Default is the Clock's stop time.
        save_first_timestep : bool, optional
            Indicates whether model output should be saved at time zero.  Default is
            True.
        output_prefix : str, optional
            String prefix for names of output netCDF files. Default is
            ``"terrainbento_output"``.
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

        self.grid.add_field(
            "node", "initial_topographic__elevation", self.z.copy()
        )

        # save output_information
        self.save_first_timestep = save_first_timestep
        self._out_file_name = output_prefix
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
                raise ValueError(
                    "Provide value for runoff_generator not valid."
                )
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
                **flow_accumulator_kwargs
            )
        else:
            self.flow_accumulator = FlowAccumulator(
                self.grid,
                flow_director=flow_director,
                depression_finder=depression_finder,
                **flow_accumulator_kwargs
            )

        ###################################################################
        # Boundary Conditions and Output Writers
        ###################################################################
        _verify_boundary_handler(boundary_handlers)
        self.boundary_handlers = boundary_handlers

        if "class" in output_writers:
            instantiated_classes = []
            for ow_class in output_writers["class"]:
                instantiated_classes.append(ow_class(self))
            output_writers["class"] = instantiated_classes

        self.output_writers = output_writers

    def _verify_fields(self, required_fields):
        """Verify all required fields are present."""
        for field in required_fields:
            if field not in self.grid.at_node:
                raise ValueError(
                    "Required field {field} not present.".format(field=field)
                )

    @property
    def model_time(self):
        """Return current time of model integration in model time units."""
        return self._model_time

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

    def write_output(self):
        """Write output to file as a netCDF.

        Filenames will have the value of ``"output_filename"`` from the
        input file or parameter dictionary as the first part of the file
        name and the model run iteration as the second part of the
        filename.
        """
        self.calculate_cumulative_change()
        filename = self._out_file_name + str(self.iteration).zfill(4) + ".nc"
        self._output_files.append(filename)
        try:
            write_raster_netcdf(
                filename, self.grid, names=self.output_fields, format="NETCDF4"
            )
        except NotImplementedError:
            graph = Graph.from_dict(
                {
                    "y_of_node": self.grid.y_of_node,
                    "x_of_node": self.grid.x_of_node,
                    "nodes_at_link": self.grid.nodes_at_link,
                }
            )

            for field_name in self.output_fields:

                graph._ds.__setitem__(
                    field_name, ("node", self.grid.at_node[field_name])
                )

            graph.to_netcdf(path=filename, mode="w", format="NETCDF4")

        self.run_output_writers()

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
        the parameter ``"step"``, and create ouput at intervals of
        ``"output_duration"``.
        """
        self._itters = []

        if self.save_first_timestep:
            self.iteration = 0
            self._itters.append(0)
            self.write_output()
        self.iteration = 1
        time_now = self._model_time
        while time_now < self.clock.stop:
            next_run_pause = min(
                time_now + self.output_interval, self.clock.stop
            )
            self.run_for(self.clock.step, next_run_pause - time_now)
            time_now = self._model_time
            self.iteration += 1
            self._itters.append(self.iteration)
            self.write_output()

        # now that the model is finished running, execute finalize.
        self.finalize()

    def _ensure_precip_runoff_are_vanilla(self, vsa_precip=False):
        """Ensure only default versions of precipitator/runoff are used.

        Some models only work when the precipitator and runoff generator
        are the default versions.
        """
        if isinstance(self.precipitator, UniformPrecipitator) is False:
            raise ValueError(
                "This model must be run with a UniformPrecipitator."
            )

        if vsa_precip is False:
            if self.precipitator._rainfall_flux != 1:
                raise ValueError(
                    "This model must use a rainfall__flux value of 1.0."
                )

        # if isinstance(self.runoff_generator, SimpleRunoff) is False:
        #     raise ValueError("This model must be run with SimpleRunoff.")

        if self.runoff_generator.runoff_proportion != 1.0:
            raise ValueError("The model must use a runoff_proportion of 1.0.")

    def run_output_writers(self):
        """Run all output writers."""
        if "class" in self.output_writers:
            for ow_class in self.output_writers["class"]:
                ow_class.run_one_step()
        if "function" in self.output_writers:
            for ow_function in self.output_writers["function"]:
                ow_function(self)

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
            self._output_files,
            concat_dim="nt",
            engine="netcdf4",
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

    def remove_output_netcdfs(self):
        """Remove all netCDF files written by a model run."""
        try:
            for f in self._output_files:
                os.remove(f)
        except WindowsError:  # pragma: no cover
            print(
                "The Windows OS is picky about file-locks and did not permit "
                "terrainbento to remove the netcdf files."
            )  # pragma: no cover


def main():  # pragma: no cover
    """Executes model."""
    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    erosion_model = ErosionModel.from_file(infile)
    erosion_model.run()


if __name__ == "__main__":
    main()
