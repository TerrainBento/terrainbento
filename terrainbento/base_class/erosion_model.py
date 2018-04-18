# -*- coding: utf-8 -*-
"""Base class for common functions of all `terrainbento`` erosion models."""

import sys
import os
import subprocess
from six import string_types
import dill
import time as tm

import numpy as np
from scipy.interpolate import interp1d

from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab import load_params
from landlab.io.netcdf import write_raster_netcdf, write_netcdf
from landlab.graph import Graph

from landlab import Component
from landlab.components import FlowAccumulator, NormalFault

from terrainbento.boundary_condition_handlers import (
                            PrecipChanger,
                            CaptureNodeBaselevelHandler,
                            ClosedNodeBaselevelHandler,
                            SingleNodeBaselevelHandler)

_SUPPORTED_BOUNDARY_HANDLERS = ['NormalFault',
                                'PrecipChanger',
                                'CaptureNodeBaselevelHandler',
                                'ClosedNodeBaselevelHandler',
                                'SingleNodeBaselevelHandler']

_HANDLER_METHODS = {'NormalFault': NormalFault,
                    'PrecipChanger': PrecipChanger,
                    'CaptureNodeBaselevelHandler': CaptureNodeBaselevelHandler,
                    'ClosedNodeBaselevelHandler': ClosedNodeBaselevelHandler,
                    'SingleNodeBaselevelHandler': SingleNodeBaselevelHandler}


class ErosionModel(object):
    """ Base class providing common functionality for ``terrainbento`` models.

    An ``ErosionModel the skeleton for the models of terrain evolution in
    ``terrainbento``. It can be initialized with an input DEM, or parameters
    used for creation of a new ``RasterModelGrid`` or ``HexModelGrid``.

    This is a base class that does not implement any processes, but rather
    simply handles I/O and setup. Derived classes are meant to include
    Landlab components to model actual erosion processes.

    It is expected that a derived model will define an ``__init__`` and a
     ``run_one_step`` method. If desired, the derived model can overwrite the
     existing ``run_for`` and ``run`` methods. 

    Methods
    -------
    read_topography
    setup_hexagonal_grid
    setup_raster_grid
    run_for
    run
    get_parameter_from_exponent
    calculate_cumulative_change
    update_boundary_conditions
    check_walltime
    write_output
    finalize
    pickle_self
    unpickle_self
    """
    def __init__(self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None):
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
            Classes used to handle
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes used to handler...

        Other Parameters
        ----------------
        pickle_name : str, optional
            Default value is 'saved_model.model'
        load_from_pickle : boolean, optional
            Default is False



        DEM_filename

        number_of_node_rows 
        number_of_node_columns
        node_spacing
        initial_elevation : float, optional
            Default value is 0.0
        add_random_noise : boolean, optional
            Default value is True.
        
        initial_noise_std : float, optional

        
        meters_to_feet : boolean, optional
            Default value is False.
        feet_to_meters : boolean, optional
            Default value is False.
            
        flow_director : str, optional
            Default is 'FlowDirectorSteepest'
            
        depression_finder : str, optional
            Default is 'DepressionFinderAndRouter'


        save_first_timestep
        outlet_id

        Returns
        -------
        ErosionModel : object
        """
        #######################################################################
        # get parameters
        #######################################################################
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

        #######################################################################
        # Get the pickled instance name.
        #######################################################################
        self.save_model_name = self.params.get('pickle_name', 'saved_model.model')
        self.load_from_pickle = self.params.get('load_from_pickle', False)

        # if pickled instance exists and should be loaded, load it.
        if (self.load_from_pickle) and (os.path.exists(self.save_model_name)):
            self.unpickle_self()

        #######################################################################
        # otherwise initialize as normal.
        #######################################################################
        else:
            # identify if initial conditions should be saved.
            # default behavior is to not save the first timestep
            self.save_first_timestep = self.params.get('save_first_timestep', False)

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
                self._starting_topography = 'inputDEM'
                (self.grid, self.z) = self.read_topography(self.params['DEM_filename'],
                                                           name='topographic__elevation',
                                                           halo=1)
                self.opt_watershed = True
            else:
                # this routine will set self.opt_watershed internally
                if self.params.get('model_grid', 'RasterModelGrid') == 'HexModelGrid':
                    self._starting_topography = 'HexModelGrid'
                    self.setup_hexagonal_grid()
                else:
                    self._starting_topography = 'RasterModelGrid'
                    self.setup_raster_grid()

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
            self.params['length_factor'] = self._length_factor

            ###################################################################
            # Boundary Conditions
            ###################################################################
            self.boundary_handler = {}

            if 'BoundaryHandlers' in self.params:
                    BoundaryHandlers = self.params['BoundaryHandlers']

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
            self.output_writers = {}
            if OutputWriters is None:
                pass
            else:
                if isinstance(OutputWriters, list):
                    for comp in OutputWriters:
                        self.setup_output_writer(comp)
                else:
                    self.setup_output_writer(OutputWriters)
                    
    def setup_boundary_handler(self, handler):
        """

        """
        if isinstance(handler, Component):
            name = handler.__name__
        else:
            name = handler
            handler = _HANDLER_METHODS[name]

        if name in _SUPPORTED_BOUNDARY_HANDLERS:

            # if unique paramters for the boundary condition handler have
            # been passed, use them.
            if name in self.params:
                handler_params = self.params[name]
                handler_params['length_factor'] = self._length_factor

            # otherwise pass all parameters
            else:
                handler_params = self.params

            # Instantiate handler
            self.boundary_handler[name] = handler(self.grid, **handler_params)

        # Raise an error if the handler is not supported.
        else:
            raise ValueError(('Only supported boundary condition handlers are '
                              'permitted. These include:'
                              '\n'.join(_SUPPORTED_BOUNDARY_HANDLERS)))
            
    def setup_output_writer(self, writer):
        """

        """
        if isinstance(writer, object):
            name = writer.__name__
            self.output_writers[name] = writer(self)
        else:
            # if a function
            pass

    def setup_hexagonal_grid(self):
        """Create hexagonal grid based on input parameters.

        Called if DEM is not used, or not found.

        Examples
        --------
        >>> params = {'model_grid' : 'HexModelGrid',
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0 }
        >>> from terrainbento import ErosionModel
        >>> em = ErosionModel(params=params)

        """

        try:
            nr = self.params['number_of_node_rows']
            nc = self.params['number_of_node_columns']
            dx = self.params['node_spacing']


        except KeyError:
            print('Warning: no DEM or grid shape specified. '
                  'Creating simple hex grid')
            nr = 8
            nc = 5
            dx = 10

        orientation = self.params.get('orientation', 'horizontal')
        shape = self.params.get('shape', 'hex')
        reorient_links = self.params.get('reorient_links', True)

        # Create grid
        from landlab import HexModelGrid
        self.grid = HexModelGrid(nr,
                                 nc,
                                 dx,
                                 shape=shape,
                                 orientation=orientation,
                                 reorient_links=reorient_links)

        # Create and initialize elevation field
        self._create_synthetic_topography()

        # Set boundary conditions
        self._setup_synthetic_boundary_conditions()
        
    def setup_raster_grid(self):
        """Create raster grid based on input parameters.

        Called if DEM is not used, or not found.

        Parameters
        ----------
        number_of_node_rows
        number_of_node_columns
        node_spacing
        outlet_id

        Examples
        --------
        >>> params = { 'number_of_node_rows' : 6,
        ...            'number_of_node_columns' : 9,
        ...            'node_spacing' : 10.0 }
        >>> from terrainbento import ErosionModel
        >>> em = ErosionModel(params=params)
        """
        try:
            nr = self.params['number_of_node_rows']
            nc = self.params['number_of_node_columns']
            dx = self.params['node_spacing']
        except KeyError:
            print('Warning: no DEM or grid shape specified. '
                  'Creating simple raster grid')
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
        
        add_noise = self.params.get('add_random_noise', True)
        init_z = self.params.get('initial_elevation', 0.0)
        init_sigma = self.params.get('initial_noise_std', 1.0)
          
        self.z = self.grid.add_zeros('node', 'topographic__elevation')
        
        if 'random_seed' in self.params:
            seed = self.params['random_seed']
        else:
            seed = 0
        np.random.seed(seed)
        
        if add_noise:
            rs = np.random.randn(len(self.grid.core_nodes))
            self.z[self.grid.core_nodes] = init_z + (init_sigma * rs)
        
    def _setup_synthetic_boundary_conditions(self):
        
        if self._starting_topography == 'HexModelGrid':
            if 'outlet_id' in self.params:
                self.opt_watershed = True
                self.outlet_node = self.params['outlet_id']
            else:
                self.opt_watershed = False
                self.outlet_node = 0
                
            
        else:  
            if 'outlet_id' in self.params:
                self.opt_watershed = True
                self.outlet_node = self.params['outlet_id']
            else:
                self.opt_watershed = False
                self.outlet_node = 0
                east_closed = north_closed = west_closed = south_closed = False
                if 'east_boundary_closed' in self.params:
                    east_closed = self.params['east_boundary_closed']
                if 'north_boundary_closed' in self.params:
                    north_closed = self.params['north_boundary_closed']
                if 'west_boundary_closed' in self.params:
                    west_closed = self.params['west_boundary_closed']
                if 'south_boundary_closed' in self.params:
                    south_closed = self.params['south_boundary_closed']
    
    
            if not self.opt_watershed:
                self.grid.set_closed_boundaries_at_grid_edges(east_closed,
                                                              north_closed,
                                                              west_closed,
                                                              south_closed)
        
    def read_topography(self, topo_file_name, name, halo):
        """Read and return topography from file.

        Parameters
        ----------
        topo_file_name
        name
        halo

        Returns
        -------
        (grid, z) : tuple
          Model grid and topographic elevation
        
        Examples
        --------

        
        """
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
            graph = Graph.from_dict({'y_of_node': self.grid.y_of_node,
               'x_of_node': self.grid.x_of_node,
               'nodes_at_link': self.grid.nodes_at_link})
            
            if field_names:
                pass
            else:
                field_names = self.grid.at_node.keys()
                
            for field_name in field_names:
                
                graph._ds.__setitem__(field_name, ('node', self.grid.at_node[field_name]))
            
            graph.to_netcdf(path=filename, mode='w', format='NETCDF4')
         
        self.run_output_writers()
        
    
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

        # now that the model is finished running, execute finalize.
        self.finalize()
        # once done, remove saved model object if it exists
        if os.path.exists(self.save_model_name):
            os.remove(self.save_model_name)
            
    def run_output_writers(self):
        """ """
        if self.output_writers is not None:
            for name in self.output_writers:
                self.output_writers[name].run_one_step()

    def update_boundary_conditions(self, dt):
        """
        Update outlet level
        """
        # Run each of the baselevel handlers.
        if self.boundary_handler is not None:
            for handler_name in self.boundary_handler:
                self.boundary_handler[handler_name].run_one_step(dt)

    def pickle_self(self):
        """Pickle model object."""
        with open(self.save_model_name, 'wb') as f:
            dill.dump(self, f)

    def unpickle_self(self):
        """Unpickle model object."""
        with open(self.save_model_name, 'rb') as f:
            model = dill.load(f)
        self.__setstate__(model)

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
