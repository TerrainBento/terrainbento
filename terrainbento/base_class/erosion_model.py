# -*- coding: utf-8 -*-
"""
Base class for common functions of terrainbento models.

"""

from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab import load_params
from landlab.io.netcdf import write_raster_netcdf, write_netcdf

from landlab.components import FlowAccumulator

import numpy as np
from scipy.interpolate import interp1d
import sys
import os
import subprocess
import dill
import time as tm
from terrainbento.boundary_condition_handlers import PrecipChanger

DAYS_PER_YEAR = 365.25

class ErosionModel(object):
    """
    Base class providing common functionality for TerrainBento models.

    An ErosionModel the skeleton for the basic models of terrain evolution in
    TerrainBento. It can be initialized with an input DEM, a or parameters
    used for creation of a new model grid.

    This is a base class that does not implement any processes, but rather
    simply handles I/O and setup. Derived classes are meant to include
    Landlab components to model actual erosion processes.

    """
    def __init__(self, input_file=None, params=None, BoundaryHandlers=None):
        """
        Parameters
        ----------
        input_file :
            df
        params :
            dfa
        BoundaryHandlers :




        Other Parameters
        ----------------
        pickle_name : str, optional
            Default value is 'saved_model.model'
        load_from_pickle : boolean, optional
            Default is False

        DEM_filename

        number_of_node_rows
        number_of_node_columns


        meters_to_feet
        feet_to_meters

        save_first_timestep
        outlet_id

        Returns
        -------
        ErosionModel : object


        """
        #######################################################################
        # Get the pickled instance name.
        #######################################################################
        self.save_model_name = self.params.get('pickle_name', 'saved_model.model')
        self.load_from_pickle = self.params.get('load_from_pickle', False)

        # if pickled instance exists and should be loaded, load it.
        if (self.load_from_pickle) and (os.path.exists(self.save_model_name)):
            with open(self.save_model_name, 'rb') as f:
                model = dill.load(f)
                self.__setstate__(model)

        #######################################################################
        # otherwise initialize as normal.
        #######################################################################
        else:
            # Import input file or parameter dictionary, checking that at least
            # one but not both were supplied.
            if input_file is None and params is None:
                raise ValueError(('ErosionModel requires one of `input_file` or '
                                  '`params` dictionary but neither were supplied.'))
            elif input_file is not None and params is not None:
                raise ValueError(('ErosionModel requires one of `input_file` or '
                                  '`params` dictionary but both were supplied.'))
            else:
                # parameter dictionary
                if input_file is None:
                    self.params = params
                # read from file.
                else:
                    self.params = load_params(input_file)


            # identify if initial conditions should be saved.
            # default behavior is to not save the first timestep
            self.save_first_timestep = self.params('save_first_timestep', False)
            self.opt_var_precip = self.params.get('opt_var_precip', False)

            # instantiate model time.
            self.model_time = 0.

            # instantiate container for computational timestep:
            self.compute_time = [tm.time()]

            # Handle option to save if walltime is to short
            self.opt_save = self.params.get('opt_save', False)

            ###################################################################
            # create topography
            ###################################################################

            # Read the topography data and create a grid
            # first, check to make sure both DEM and node-rows are not both
            # specified.
            if ((self.params.get('number_of_node_rows') is not None) and
                (self.params.get('DEM_filename') is not None)):
                raise ValueError('Both a DEM filename and number_of_node_rows have '
                                 'been specified.')

            if 'DEM_filename' in self.params:
                (self.grid, self.z) = self.read_topography(self.params['DEM_filename'],
                                                           name='topographic__elevation',
                                                           halo=1)
                self.opt_watershed = True
                self._starting_topography = 'inputDEM'

            else:
                # this routine will set self.opt_watershed internally
                if self.params.get('model_grid', 'RasterModelGrid') == 'HexModelGrid':
                    self.setup_hexagonal_grid(self.params)
                    self._starting_topography = 'HexModelGrid'
                else:
                    self.setup_rectangular_grid(self.params)
                    self._starting_topography = 'RasterModelGrid'


            # Set DEM boundaries
            if self.opt_watershed:
                try:
                    self.outlet_node = self.params['outlet_id']
                    self.grid.set_watershed_boundary_condition_outlet_id(self.outlet_node,
                                                                         self.z,
                                                                         nodata_value=-9999)
                except:
                    self.outlet_node = self.grid.set_watershed_boundary_condition(self.z,
                                                                                  nodata_value=-9999,
                                                                                  return_outlet_id=True)

            # Add fields for initial topography and cumulative erosion depth
            z0 = self.grid.add_zeros('node', 'initial_topographic__elevation')
            z0[:] = self.z  # keep a copy of starting elevation
            self.grid.add_zeros('node', 'cumulative_erosion__depth')

            # identify which nodes are data nodes:
            self.data_nodes = self.grid.at_node['topographic__elevation']!=-9999.

            ###################################################################
            # instantiate flow direction and accumulation
            ###################################################################
            # get flow direction, and depression finding options
            self.flow_director = params.get('flow_director', 'FlowDirectorSteepest')
            self.depression_finder = params.get('depression_finder', None)

            # Instantiate a FlowAccumulator, if DepressionFinder is provided
            # AND director = Steepest, then we need routing to be D4,
            # otherwise, just passing params should be sufficient.
            if ((self.depression_finder is not None) and
                (self.flow_director == 'FlowDirectorSteepest')):
                self.flow_router = FlowAccumulator(self.grid,
                                                   routing = 'D4',
                                                   **self.params)
            else:
                self.flow_router = FlowAccumulator(self.grid, **self.params)

            ###################################################################
            # get internal length scale adjustement
            ###################################################################
            feet_to_meters = self.params.get('feet_to_meters', False)
            meters_to_feet = self.params.get('meters_to_feet', False)
            if feet_to_meters and meters_to_feet:
                raise ValueError('Both "feet_to_meters" and "meters_to_feet" are'
                                 'set as True. This is not realistic.')
            else:
                if feet_to_meters:
                    self._length_factor = 1.0/3.28084
                elif meters_to_feet:
                    self._length_factor = 3.28084
                else:
                    self._length_factor = 1.0

            ###################################################################
            # Boundary Conditions
            ###################################################################
            # Read and remember baselevel control param, if present
            self.outlet_lowering_rate = self.params.get('outlet_lowering_rate',  0.0)
            try:
                file_name = self.params['outlet_lowering_file_path']

                modern_outlet_elevation = self.params['modern_outlet_elevation']
                postglacial_outlet_elevation = self.z[self.outlet_node]

                elev_change_df = np.loadtxt(file_name, skiprows=1, delimiter =',')
                time = elev_change_df[:, 0]
                elev_change = elev_change_df[:, 1]

                scaling_factor = np.abs(postglacial_outlet_elevation-modern_outlet_elevation)/np.abs(elev_change[0]-elev_change[-1])

                outlet_elevation = (scaling_factor*elev_change_df[:, 1]) + postglacial_outlet_elevation

                self.outlet_elevation_obj = interp1d(time, outlet_elevation)

            except KeyError:
                self.outlet_elevation_obj = None

            # BoundaryHandlers
            if BoundaryHandlers is None:
                self.boundary_handler = None
            else:
                self.boundary_handler = {}
                if isinstance(BoundaryHandlers, list):
                    for comp in BoundaryHandlers:
                        self.boundary_handler[comp.__name__] = comp(self.grid, **self.params)
                else:
                    self.boundary_handler[BoundaryHandlers.__name__] = BoundaryHandlers(self.grid, **self.params)

    def setup_hexagonal_grid(self, params):
        """
        """
        """Create hexagonal grid based on input parameters.

        Called if DEM is not used, or not found.

        Examples
        --------
        >>> params = {'model_grid' : 'HexModelGrid',
        ...           'base_num_rows' : 6,
        ...           'base_num_cols' : 9,
        ...           'dx' : 10.0 }
        >>> from terrainbento import ErosionModel
        >>> em = ErosionModel(params=params)
        """

        try:
            nr = params['number_of_node_rows']
            nc = params['number_of_node_columns']
            dx = params['node_spacing']


        except KeyError:
            print('Warning: no DEM or grid shape specified. '
                  'Creating simple hex grid')
            nr = 8
            nc = 5
            dx = 10

        orientation = params.get('orientation', 'horizontal')
        shape = params.get('shape', 'hex')
        reorient_links = params.get('reorient_links', True)

        if 'outlet_id' in params:
            self.opt_watershed = True
            self.outlet_node = params['outlet_id']
        else:
            self.opt_watershed = False
            self.outlet_node = 0

        # Create grid
        from landlab import HexModelGrid
        self.grid = HexModelGrid(nr,
                                 nc,
                                 dx,
                                 shape=shape,
                                 orientation=orientation,
                                 reorient_links=reorient_links)

        # Create and initialize elevation field
        self.z = self.grid.add_zeros('node', 'topographic__elevation')
        if 'random_seed' in params:
            seed = params['random_seed']
        else:
            seed = 0
        np.random.seed(seed)
        rs = np.random.rand(len(self.grid.core_nodes))
        self.z[self.grid.core_nodes] = rs

        # Set boundary conditions

    def setup_rectangular_grid(self, params):
        """Create rectangular grid based on input parameters.

        Called if DEM is not used, or not found.

        Examples
        --------
        >>> params = { 'number_of_node_rows' : 6,
        ...            'number_of_node_columns' : 9,
        ...            'node_spacing' : 10.0 }
        >>> from terrainbento import ErosionModel
        >>> em = ErosionModel(params=params)
        """
        try:
            nr = params['number_of_node_rows']
            nc = params['number_of_node_columns']
            dx = params['node_spacing']
        except KeyError:
            print('Warning: no DEM or grid shape specified. '
                  'Creating simple raster grid')
            nr = 4
            nc = 5
            dx = 1.0

        if 'outlet_id' in params:
            self.opt_watershed = True
            self.outlet_node = params['outlet_id']
        else:
            self.opt_watershed = False
            self.outlet_node = 0
            east_closed = north_closed = west_closed = south_closed = False
            if 'east_boundary_closed' in params:
                east_closed = params['east_boundary_closed']
            if 'north_boundary_closed' in params:
                north_closed = params['north_boundary_closed']
            if 'west_boundary_closed' in params:
                west_closed = params['west_boundary_closed']
            if 'south_boundary_closed' in params:
                south_closed = params['south_boundary_closed']

        # Create grid
        from landlab import RasterModelGrid
        self.grid = RasterModelGrid((nr, nc), dx)

        # Create and initialize elevation field
        self.z = self.grid.add_zeros('node', 'topographic__elevation')
        if 'random_seed' in params:
            seed = params['random_seed']
        else:
            seed = 0
        np.random.seed(seed)
        rs = np.random.rand(len(self.grid.core_nodes))
        self.z[self.grid.core_nodes] = rs

        # Set boundary conditions
        if not self.opt_watershed:
            self.grid.set_closed_boundaries_at_grid_edges(east_closed,
                                                          north_closed,
                                                          west_closed,
                                                          south_closed)

    def read_topography(self, topo_file_name, name, halo):
        """Read and return topography from file, as a Landlab grid and field."""
        try:
            (grid, z) = read_esri_ascii(topo_file_name,
                                        name=name,
                                        halo=halo)
        except:
            grid = read_netcdf(topo_file_name)
            z = grid.at_node[name]
        return (grid, z)

    def get_parameter_from_exponent(self, param_name, raise_error=True):
        """Return absolute parameter value from provided exponent.
        """
        if (param_name in self.params) and (param_name+'_exp' in self.params):
            raise ValueError('Parameter file includes both absolute value and'
                             'exponent version of:'+ param_name)

        if (param_name in self.params) and (param_name+'_exp' not in self.params):
            param = self.params[param_name]
        elif (param_name not in self.params) and (param_name+'_exp' in self.params):
            param = 10.**float(self.params[param_name+'_exp'])
        else:
            if raise_error:
                raise ValueError('Parameter file includes neither absolute'
                                 'value or exponent version of:'+ param_name)
            else:
                param = None
        return param

    def __setstate__(self, state_dict):
        """Get ErosionModel state from pickled state_dict."""
        random_state = state_dict.pop('random_state')
        np.random.set_state(random_state)

        if state_dict['outlet_elevation_obj'] == None:
            pass
        else:
            outlet_elev = state_dict.pop('outlet_elevation_obj')
            state_dict['outlet_elevation_obj'] = interp1d(outlet_elev['x'], outlet_elev['y'])

        self.__dict__ = state_dict

    def __getstate__(self):
        """Set ErosionModel state for pickling."""
        state_dict = self.__dict__
        state_dict['random_state'] = np.random.get_state()

        if self.outlet_elevation_obj == None:
            pass
        else:
            x = self.outlet_elevation_obj.x
            y = self.outlet_elevation_obj.y
            state_dict['outlet_elevation_obj'] = {'x': x, 'y': y}
        return state_dict

    def calculate_cumulative_change(self):
        """Calculate cumulative node-by-node changes in elevation.

        Store result in grid field.
        """
        self.grid.at_node['cumulative_erosion__depth'] = \
            self.grid.at_node['topographic__elevation'] - \
            self.grid.at_node['initial_topographic__elevation']
        max_cc = np.amax(self.grid.at_node['cumulative_erosion__depth'])
        min_cc = np.amin(self.grid.at_node['cumulative_erosion__depth'])
        print('Maximum cumulative topo change:')
        print(max_cc)
        print('Minimum cumulative topo change:')
        print(min_cc)


    def write_output(self, params, field_names=None):
        """Write output to file (currently netCDF)."""

        # Exclude fields with int64 (incompatible with netCDF3)
        if field_names is None:
            field_names = []
            for field in self.grid.at_node:
                if type(self.grid.at_node[field][0]) is not np.int64:
                    field_names.append(field)

        self.calculate_cumulative_change()
        filename = self.params['output_filename'] + str(self.iteration).zfill(4) \
                    + '.nc'
        try:
            write_raster_netcdf(filename, self.grid, names=field_names, format='NETCDF4')
        except NotImplementedError:
            #write_netcdf(filename, self.grid, names=field_names, format='NETCDF4')
            pass

    def run_one_step(self, dt):
        """
        Run each component for one time step.

        This base-class method does nothing. Derived classes should override
        it to run each component in turn for a time period dt.
        """
        pass

    def finalize(self):
        """
        Finalize model

        This base-class method does nothing. Derived classes can override
        it to run any required finalizations steps.
        """
        pass

    def run_for(self, dt, runtime):
        """
        Run model without interruption for a specified time period.
        """
        elapsed_time = 0.
        keep_running = True
        while keep_running:
            if elapsed_time+dt >= runtime:
                dt = runtime-elapsed_time
                keep_running = False
            self.run_one_step(dt)
            elapsed_time += dt

    def run(self, output_fields=None):
        """
        Run the model until complete.
        """
        if self.save_first_timestep:
            self.iteration = 0
            self.write_output(self.params, field_names=output_fields)
        total_run_duration = self.params['run_duration']
        output_interval = self.params['output_interval']
        self.iteration = 1
        time_now = self.model_time
        while time_now < total_run_duration:
            next_run_pause = min(time_now + output_interval, total_run_duration)
            self.run_for(self.params['dt'], next_run_pause - time_now)
            time_now = next_run_pause
            self.write_output(self.params, field_names=output_fields)
            self.iteration += 1

        self.finalize()
        # once done, remove saved model object if it exists
        if os.path.exists(self.save_model_name):
            os.remove(self.save_model_name)

    def update_outlet(self, dt):
        """
        Update outlet level
        """
        # determine which nodes to lower
        if self.opt_watershed:
            # if we are dealing with a watershed, then lower only the outlet node
            nodes_to_lower = self.outlet_node
        else:
            # if we are dealing with a rectangular grid, then raise only the core nodes
            nodes_to_lower = self.grid.status_at_node == 0

        # next, lower the correct nodes the desired amount

        # first, if we do not have an outlet elevation object
        if self.outlet_elevation_obj is None:

            # calculate lowering amount and subtract
            if self.opt_watershed:
                self.z[nodes_to_lower] -= self.outlet_lowering_rate * dt
            else:
                # if this is not a watershed, we are raising the core nodes
                self.z[nodes_to_lower] += self.outlet_lowering_rate * dt

            # if bedrock_elevation exists as a field, lower it also
            if 'bedrock__elevation' in self.grid.at_node.keys():
                if self.opt_watershed:
                    self.grid.at_node['bedrock__elevation'][nodes_to_lower] -= self.outlet_lowering_rate * dt
                else:
                    self.grid.at_node['bedrock__elevation'][nodes_to_lower] += self.outlet_lowering_rate * dt
        # if there is an outlet elevation object
        else:
            # if bedrock_elevation exists as a field, lower it also
            # calcuate the topographic change required to match the current time's value for
            # outlet elevation. This must be done in case bedrock elevation exists, and must
            # be done before the topography is lowered
            if 'bedrock__elevation' in self.grid.at_node.keys():
                topo_change = self.z[nodes_to_lower] - self.outlet_elevation_obj(self.model_time)
                self.grid.at_node['bedrock__elevation'][nodes_to_lower] -= topo_change

            # lower topography
            self.z[nodes_to_lower] = self.outlet_elevation_obj(self.model_time)

        # Run each of the baselevel handlers.
        if self.boundary_handler is not None:
            for handler_name in self.boundary_handler:
                self.boundary_handler[handler_name].run_one_step(dt)

    def pickle_self(self):
        """Pickle model object."""
        with open(self.save_model_name, 'wb') as f:
                    dill.dump(self, f)

    def check_walltime(self, wall_threshold=0, dynamic_cut_off_time=False, cut_off_time=0):
        """Check walltime and save model out if near end of time."""
        # check walltime

        # format is days-hours:minutes:seconds
        if dynamic_cut_off_time:
            self.compute_time.append(tm.time())
            mean_time_diffs = np.mean(np.diff(np.asarray(self.compute_time)))/60. # in minutes
            cut_off_time = mean_time_diffs + wall_threshold
        else:
            pass # cut off time is 0

        try:
            output,error = subprocess.Popen(['squeue',
                                              '--job='+os.environ['SLURM_JOB_ID'], '--format=%.10L'],
                                             stdout = subprocess.PIPE,
                                             stderr = subprocess.PIPE).communicate()
            time_left = output.strip().split(' ')[-1]

            if len(time_left.split('-')) == 1:
                days_left = 0
            else:
                days_left = time_left.split('-')[0]

            try:
                minutes_left = int(time_left.split(':')[-2])
            except IndexError:
                minutes_left = 0

            try:
                hours_left = int(time_left.split(':')[-3])
            except IndexError:
                hours_left = 0

            remaining_time = (days_left*60*60) + (60*hours_left) + minutes_left

            if self.opt_save:
                if remaining_time < cut_off_time:
                    # pickle self
                    self.pickle_self()
                    # exit program
                    sys.exit()

        except KeyError:
            pass


def main():
    """Executes model."""
    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    erosion_model = ErosionModel(input_file=infile)
    erosion_model.run()


if __name__ == '__main__':
    main()
