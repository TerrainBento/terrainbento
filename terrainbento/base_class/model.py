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


class Model(object):
    """Defines the base model class from which terrainbento models inherit.

    **Base model class methods**

    .. autosummary::
        :toctree: generated/

        ~terrainbento.base_class.model.Model.from_path
        ~terrainbento.base_class.model.Model.name
        ~terrainbento.base_class.model.Model.units
        ~terrainbento.base_class.model.Model.definitions
        ~terrainbento.base_class.model.Model.input_var_names
        ~terrainbento.base_class.model.Model.output_var_names
        ~terrainbento.base_class.model.Model.optional_var_names
        ~terrainbento.base_class.model.Model.var_type
        ~terrainbento.base_class.model.Model.var_units
        ~terrainbento.base_class.model.Model.var_definition
        ~terrainbento.base_class.model.Model.var_mapping
        ~terrainbento.base_class.model.Model.var_loc
        ~terrainbento.base_class.model.Model.var_help
        ~terrainbento.base_class.model.Model.initialize_output_fields
        ~terrainbento.base_class.model.Model.initialize_optional_output_fields
        ~terrainbento.base_class.model.Model.shape
        ~terrainbento.base_class.model.Model.grid
        ~terrainbento.base_class.model.Model.coords
        ~terrainbento.base_class.model.Model.imshow
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
        self.grid = grid
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
            return cls._var_type[name]
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
        return tuple(self._var_units.items())

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
        return cls._var_units[name]

    @classproperty
    @classmethod
    def definitions(cls):
        """Get a description of each field.

        Returns
        -------
        tuple of (*name*, *description*)
            A description of each field.
        """
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
            init_vals = self.grid.zeros(grp, dtype=type_in)
            units_in = self.var_units(field_to_set)
            self.grid.add_field(
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
        return self.grid._shape

    @property
    def grid(self):
        """Return the grid attached to the component."""
        return self._grid

    @property
    def coords(self):
        """Return the coordinates of nodes on grid attached to the
        component."""
        return (self.grid.node_x, self.grid.node_y)
