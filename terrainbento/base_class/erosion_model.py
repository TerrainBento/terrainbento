# coding: utf8
# !/usr/env/python
"""Base class for common functions of all terrainbento erosion models.

The **ErosionModel** is a base class that contains all of the
functionality shared by the terrainbento models.
"""

import os
import sys
import time as tm
from types import FunctionType

import dask
import numpy as np
import six
import xarray as xr
import yaml

from landlab import CLOSED_BOUNDARY, ModelGrid, create_grid, load_params
from landlab.components import FlowAccumulator, NormalFault
from landlab.graph import Graph
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf, write_raster_netcdf
from terrainbento import Clock
from terrainbento.boundary_condition_handlers import (
    CaptureNodeBaselevelHandler,
    GenericFuncBaselevelHandler,
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    SingleNodeBaselevelHandler,
)

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


def _setup_boundary_handlers(grid, name, params):
    """Setup BoundaryHandlers for use by a terrainbento model.

    A boundary condition handler is a class with a **run_one_step** method
    that takes the parameter ``step``. Permitted boundary condition handlers
    include the Landlab Component ``NormalFault`` as well as the following
    options from terrainbento: **PrecipChanger**,
    **CaptureNodeBaselevelHandler**, **NotCoreNodeBaselevelHandler**,
    **SingleNodeBaselevelHandler**.

    Parameters
    ----------
    handler : str
        Name of a supported boundary condition handler.
    """
    if name in _SUPPORTED_BOUNDARY_HANDLERS:
        # Instantiate handler
        handler_func = _HANDLER_METHODS[name]
        boundary_handler = handler_func(grid, **params)
    # Raise an error if the handler is not supported.
    else:
        raise ValueError(
            (
                "Only supported boundary condition handlers are "
                "permitted. These include:"
                "\n".join(_SUPPORTED_BOUNDARY_HANDLERS)
            )
        )
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
    """

    @classmethod
    def from_file(cls, filename):
        """
        model = ErosionModel.from_file("file.yaml")
        """
        with open(filename, "r") as f:
            dict = yaml.load(f)
        return cls.from_dict(dict)

    @classmethod
    def from_dict(cls, params, outputwriters=None):
        """
        model = ErosionModel.from_dict(dict-like)
        """
        cls._validate(params)

        grid = create_grid(**params.pop("grid"))
        clock = Clock.from_dict(**params.pop("clock"))
        boundary_handlers = params.pop("boundary_handlers", {})
        bh_dict = {}
        for name in boundary_handlers:
            bh_params = boundary_handlers[name]
            bh_dict[name] = _setup_boundary_handlers(grid, name, bh_params)

        return cls(clock, grid, bh_dict, outputwriters, **params)

    @classmethod
    def _validate(cls, params):
        """Make sure necessary things for a model grid and a clock are here."""
        if "grid" not in params:
            msg = ""
            raise ValueError(msg)
        if "clock" not in params:
            msg = ""
            raise ValueError(msg)

    def __init__(
        self,
        clock,
        grid,
        precipitator=None,
        runoff_generator=None,
        boundary_handlers={},
        output_writers=None,
        flow_director="FlowDirectorSteepest",
        depression_finder=None,
        output_interval=None,
        save_first_timestep=True,
        output_prefix="terrainbento_output",
        fields=["topographic__elevation"],
        **kwargs
    ):
        """
        Parameters
        ----------
        clock : terrainbento Clock instance
        grid : landlab model grid instance
            Correct fields must be created.
        precipitator : terrainbento precipitator, optional

        runoff_generator : terrainbento runoff_generator, optional


        boundary_handlers : dictionary, optional
            terrainbento provides the ability for an arbitrary number of boundary
            condition handler classes to operate on the model grid each time step in order
            to handle time-variable boundary conditions such as: changing a watershed
            outlet elevation, modifying precipitation parameters through time, or
            simulating external drainage capture.
            Strings containing the names of classes used to handle boundary conditions.
            Valid options are currently: "NormalFault", "PrecipChanger",
            "CaptureNodeBaselevelHandler", "NotCoreNodeBaselevelHandler", and
            "SingleNodeBaselevelHandler". These BoundaryHandlers are instantiated with
            the entire parameter set unless there is an entry in the parameter
            dictionary with the name of the boundary handler that contains its own
            parameter dictionary. If this is the case, the handler-specific dictionary
            is passed to instantiate the boundary handler.
        outputwriters : class, function, or list, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).
        flow_director : str, optional
            String name of a Landlab FlowDirector. All options that the Landlab
            FlowAccumulator is compatible with are permitted. Default is
            "FlowDirectorSteepest".
        depression_finder : str, optional
            String name of a Landlab depression finder. Default is None.
        save_first_timestep : bool, optional
            Indicates whether model output should be saved at time zero.  Default is
            True.
        output_filename : str, optional
            String prefix for names of output netCDF files. Default is
            ``"terrainbento_output"``.
        output_interval : float, optional
            Default is the Clock's stop time.
        **kwargs :
            Any kwargs to pass to the FlowAccumulator.

        Returns
        -------
        ErosionModel: object

        Examples
        --------
        This model is a base class and is not designed to be run on its own. We
        recommend that you look at the terrainbento tutorials for examples of
        usage.
        """
        # type checking
        if isinstance(clock, Clock) is False:
            raise ValueError("Provided Clock is not valid.")

        if isinstance(grid, ModelGrid) is False:
            raise ValueError("Provided Grid is not valid.")

        # save the grid, clock, and parameters.
        self.grid = grid
        self.clock = clock

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
        self._model_time = 0.

        # instantiate container for computational timestep:
        self._compute_time = [tm.time()]

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
                self.grid, routing="D4", **kwargs
            )
        else:
            self.flow_accumulator = FlowAccumulator(self.grid, **kwargs)

        ###################################################################
        # Boundary Conditions and Output Writers
        ###################################################################
        self.boundary_handlers = boundary_handlers
        self.output_writers = output_writers

        if len(kwargs) > 0:
            msg = ""
            raise ValueError(kwargs)

    def _verify_fields(self, required_fields):
        """"""
        for field in required_fields:
            if field not in self.grid.at_node:
                raise ValueError

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
        """"""
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
        elapsed_time = 0.
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

    def run_output_writers(self):
        """Run all output writers."""
        if self.output_writers is not None:
            for name in self.output_writers["class"]:
                self.output_writers["class"][name].run_one_step()
            for function in self.output_writers["function"]:
                function(self)

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
        ds.set_coords(["x", "y", "time"], inplace=True)

        # rename dimensions
        ds.rename(name_dict={"ni": "x", "nj": "y", "nt": "time"}, inplace=True)

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
        ds = self.to_xarray_dataset(time_unit=time_unit, space_unit=space_unit)
        ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4")
        ds.close()

    def remove_output_netcdfs(self):
        """Remove all netCDF files written by a model run."""
        for f in self._output_files:
            os.remove(f)


def main():  # pragma: no cover
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
