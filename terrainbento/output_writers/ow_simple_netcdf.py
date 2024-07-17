#!/usr/bin/env python3

import os.path

from landlab import RasterModelGrid
from landlab.io.netcdf import to_netcdf, write_raster_netcdf

from terrainbento.output_writers.static_interval_writer import (
    StaticIntervalOutputWriter,
)


class OWSimpleNetCDF(StaticIntervalOutputWriter):
    def __init__(
        self,
        model,
        output_fields,
        name="simple-netCDF",
        **static_interval_kwargs,
    ):
        """A simple output writer which generates netCDF files at uniform
        intervals. Mimics the built-in netCDF writing code in older versions of
        terrainbento.

        Parameters
        ----------
        model : a terrainbento ErosionModel instance

        output_fields : model grid field name
            The grid field to be written to file.

        name : string, optional
            The name of the output writer used when generating output
            filenames. Defaults to 'simple-netCDF'

        static_interval_kwargs : keyword args, optional
            Keyword arguments that will be passed directly to
            StaticIntervalOutputWriter. These include:

                * intervals : float, list of floats, defaults to model duration
                * intervals_repeat : bool, defaults to False
                * times : list of floats, defaults to clock stop time
                * add_id : bool, defaults to True
                * save_first_timestep : bool, defaults to False
                * save_last_timestep : bool, defaults to True
                * output_dir : string, defaults to './output'

            Please see
            :py:class:`StaticIntervalOutputWriter` and
            :py:class:`GenericOutputWriter` for
            more detail.

        Returns
        -------
        OWSimpleNetCDF: object

        """

        super().__init__(model, name=name, **static_interval_kwargs)

        self.output_fields = output_fields

    def run_one_step(self):
        """Write output to file as a netCDF."""
        filename_prefix = self.filename_prefix
        filename = f"{filename_prefix}.nc"
        filepath = os.path.join(self.output_dir, filename)

        grid = self.model.grid
        if isinstance(grid, RasterModelGrid):
            write_raster_netcdf(
                filepath, grid, names=self.output_fields, format="NETCDF4"
            )
        else:
            to_netcdf(grid, filepath, format="NETCDF4")

        self.register_output_filepath(filepath)
