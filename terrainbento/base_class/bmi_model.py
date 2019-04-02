#! /usr/bin/env python
"""Defines the base component class from which terrainbento models inherit."""

import inspect
import os
import textwrap
import warnings

_VAR_HELP_MESSAGE = """
name: {name}
description:
{desc}
units: {units}
at: {loc}
intent: {intent}
"""


class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


class BmiModel(object):
    """Defines the base model class from which terrainbento models inherit.

    This class, ``BmiModel`` allows for all terrainbento models to have a BMI
    compatible with the Community Surface Modeling System PyMT package.

    todo: add CSDMS url link.

    **Base model class methods**

    .. autosummary::
        :toctree: generated/

        ~terrainbento.base_class.bmi_model.Model.from_path
        ~terrainbento.base_class.bmi_model.Model.name
        ~terrainbento.base_class.bmi_model.Model.units
        ~terrainbento.base_class.bmi_model.Model.definitions
        ~terrainbento.base_class.bmi_model.Model.input_var_names
        ~terrainbento.base_class.bmi_model.Model.output_var_names
        ~terrainbento.base_class.bmi_model.Model.var_type
        ~terrainbento.base_class.bmi_model.Model.var_units
        ~terrainbento.base_class.bmi_model.Model.var_definition
        ~terrainbento.base_class.bmi_model.Model.var_mapping
        ~terrainbento.base_class.bmi_model.Model.var_loc
        ~terrainbento.base_class.bmi_model.Model.var_help
        ~terrainbento.base_class.bmi_model.Model.initialize_output_fields
        ~terrainbento.base_class.bmi_model.Model.shape
        ~terrainbento.base_class.bmi_model.Model.grid
        ~terrainbento.base_class.bmi_model.Model.coords

        todo add the others
        todo add ones in BMI


    """

    _name = "Model"

    _input_var_names = set()

    _output_var_names = set()

    _var_info = dict()

    _param_info = dict()

    def __new__(cls, *args, **kwds):
        return object.__new__(cls)

    def __init__(self, clock, grid):
        # save the grid, clock, and parameters.
        self._grid = grid
        self.clock = clock

    @classproperty
    @classmethod
    def input_var_names(cls):
        """Names of fields that are used by the component.

        Returns
        -------
        tuple of str
            Tuple of field names.
        """
        return tuple(cls._input_var_names)

    @classproperty
    @classmethod
    def output_var_names(self):
        """Names of fields that are provided by the component.

        Returns
        -------
        tuple of str
            Tuple of field names.
        """
        return tuple(self._output_var_names)

    @classmethod
    def var_type(cls, name):
        """Returns the dtype of a field (float, int, bool, str...), if
        declared. Default is float.

        Parameters
        ----------
        name : str
            A field name.

        Returns
        -------
        dtype
            The dtype of the field.
        """
        try:
            return cls._varinfo[name]["type"]
        except AttributeError:
            return float

    @classproperty
    @classmethod
    def name(self):
        """Name of the component.

        Returns
        -------
        str
            Component name.
        """
        return self._name

    @classproperty
    @classmethod
    def units(self):
        """Get the units for all field values.

        Returns
        -------
        tuple or str
            Units for each field.
        """
        units = [info["units"] for info in self._var_info]
        return tuple(units)

    @classmethod
    def var_units(cls, name):
        """Get the units of a particular field.

        Parameters
        ----------
        name : str
            A field name.

        Returns
        -------
        str
            Units for the given field.
        """
        return cls._var_info[name]["units"]

    @classproperty
    @classmethod
    def definitions(cls):
        """Get a description of each field.

        Returns
        -------
        tuple of (*name*, *description*)
            A description of each field.
        """
        todo. this needs uupdating.
        return tuple(cls._var_doc.items())

    @classmethod
    def var_definition(cls, name):
        """Get a description of a particular field.

        Parameters
        ----------
        name : str
            A field name.

        Returns
        -------
        tuple of (*name*, *description*)
            A description of each field.
        """
        this needs updating
        return cls._var_doc[name]

    @classmethod
    def var_help(cls, name):
        """Print a help message for a particular field.

        Parameters
        ----------
        name : str
            A field name.
        """
        desc = os.linesep.join(
            textwrap.wrap(
                cls._var_doc[name], initial_indent="  ", subsequent_indent="  "
            )
        )
        units = cls._var_units[name]
        loc = cls._var_mapping[name]

        intent = ""
        if name in cls._input_var_names:
            intent = "in"
        if name in cls._output_var_names:
            intent += "out"

        help = _VAR_HELP_MESSAGE.format(
            name=name, desc=desc, units=units, loc=loc, intent=intent
        )

        print(help.strip())

    @classproperty
    @classmethod
    def var_mapping(self):
        """Location where variables are defined.

        Returns
        -------
        tuple of (name, location)
            Tuple of variable name and location ('node', 'link', etc.) pairs.
        """
        return tuple(self._var_mapping.items())

    @classmethod
    def var_loc(cls, name):
        """Location where a particular variable is defined.

        Parameters
        ----------
        name : str
            A field name.

        Returns
        -------
        str
            The location ('node', 'link', etc.) where a variable is defined.
        """
        return cls._var_mapping[name]

    def param...:
    def initialize_output_fields(self):
        """Create fields for a component based on its input and output var
        names.

        This method will create new fields (without overwrite) for any
        fields output by, but not supplied to, the component. New fields
        are initialized to zero. Ignores optional fields, if specified
        by _optional_var_names. New fields are created as arrays of
        floats, unless the component also contains the specifying
        property _var_type.
        """
        for field_to_set in (
            set(self.output_var_names)
            - set(self.input_var_names)
        ):
            grp = self.var_loc(field_to_set)
            type_in = self.var_type(field_to_set)
            init_vals = self._grid.zeros(grp, dtype=type_in)
            units_in = self.var_units(field_to_set)
            self._grid.add_field(
                grp,
                field_to_set,
                init_vals,
                units=units_in,
                copy=False,
                noclobber=True,
            )

    @property
    def shape(self):
        """Return the grid shape attached to the component, if defined."""
        return self._grid._shape

    @property
    def grid(self):
        """Return the grid attached to the component."""
        return self._grid

    @property
    def coords(self):
        """Return the coordinates of nodes on grid attached to the component."""
        return (self._grid.node_x, self._grid.node_y)

    def get_component_name(self):
        """Name of the component."""
        return self._cls.name

    def get_input_var_names(self):
        """Names of the input exchange items."""
        return self._cls.input_var_names

    def get_output_var_names(self):
        """Names of the output exchange items."""
        return self._cls.output_var_names

    def get_current_time(self):
        """Current component time."""
        return self._base.clock.time

    def get_end_time(self):
        """Stop time for the component."""
        return self._base.clock.stop

    def get_start_time(self):
        """Start time of the component."""
        return self._base.clock.start

    def get_time_step(self):
        """Component time step."""
        return self._base.clock.step

    def get_time_units(self):
        """Time units used by the component."""
        return spam

    def initialize(self, fname):
        """Initialize the component from a file.

        BMI-wrapped terrainbento models use input files in YAML format.

        Component-specific parameters are listed at the top level,
        followed by grid and then time information. An example input
        file looks like::

         clock:
             start: 0
             stop: 100.
             step: 2.
         grid:
             type: raster
             shape: [20, 40]
             spacing: [1000., 2000.]

        In this case, a `RasterModelGrid` is created (with the given shape
        and spacing) and passed to the underlying landlab component. The
        `eet=15000.` is also given to the component but as a keyword
        parameter. The BMI clock is initialized with the given parameters.

        Parameters
        ----------
        fname : str or file_like
         YAML-formatted input file for the component.
        """
        self._base = self._cls.from_file(fname)

    def update(self):
        """Update the component one time step."""
        if hasattr(self._base, "update"):
            self._base.update()
        self._clock.advance()

    def update_frac(self, frac):
        """Update the component a fraction of a time step."""
        time_step = self.get_time_step()
        self._clock.step = time_step * frac
        self.update()
        self._clock.step = time_step

    def update_until(self, then):
        """Update the component until a given time."""
        n_steps = (then - self.get_current_time()) / self.get_time_step()
        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

    def finalize(self):
        """Clean-up the component."""
        pass

    def get_var_grid(self, name):
        """Get the grid id for a variable."""
        return 0

    def get_var_itemsize(self, name):
        """Get the size of elements of a variable."""
        return np.dtype("float").itemsize

    def get_var_nbytes(self, name):
        """Get the total number of bytes used by a variable."""
        return self.get_itemsize(name) * self._base.grid.number_of_nodes

    def get_var_type(self, name):
        """Get the data type for a variable."""
        return str(np.dtype("float"))

    def get_var_units(self, name):
        """Get the unit used by a variable."""
        return self._cls.var_units(name)

    def get_value_ref(self, name):
        """Get a reference to a variable's data."""
        return self._base.grid.at_node[name]

    def get_value(self, name):
        """Get a copy of a variable's data."""
        return self._base.grid.at_node[name].copy()

    def set_value(self, name, vals):
        """Set the values of a variable."""
        if name in self.get_input_var_names():
            if name in self._base.grid.at_node:
                self._base.grid.at_node[name][:] = vals.flat
            else:
                self._base.grid.at_node[name] = vals
        else:
            raise KeyError("{name} is not an input item".format(name=name))

    def get_grid_origin(self, gid):
        """Get the origin for a structured grid."""
        return (self._base.grid.node_y[0], self._base.grid.node_x[0])

    def get_grid_rank(self, gid):
        """Get the number of dimensions of a grid."""
        return 2

    def get_grid_shape(self, gid):
        """Get the shape of a structured grid."""
        return (
         self._base.grid.number_of_node_rows,
         self._base.grid.number_of_node_columns,
        )

    def get_grid_spacing(self, gid):
        """Get the row and column spacing of a structured grid."""
        return (self._base.grid.dy, self._base.grid.dx)

        def get_grid_type(self, gid):
            """Get the type of grid."""
            return
