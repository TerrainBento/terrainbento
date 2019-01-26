"""
========================================================================================
Wrap landlab component with the Basic Modeling Interface (:mod:`landlab.bmi.bmi_bridge`)
========================================================================================

.. sectionauthor:: Eric Hutton

Function reference
------------------

The `wrap_as_bmi` function wraps a landlab component class so that it
exposes a Basic Modelling Interface.

"""
import os

import numpy as np
import yaml

from ..core.model_component import Component
from ..grid import RasterModelGrid

def wrap_as_bmi(cls):
    """Wrap a landlab class so it exposes a BMI.

    Parameters
    ----------
    cls : class
        A landlab class that inherits from `Component`.

    Returns
    -------
    class
        A wrapped class that exposes a BMI.

    Examples
    --------
    >>> from landlab.bmi import wrap_as_bmi
    >>> from landlab.components.flexure import Flexure

    >>> BmiFlexure = wrap_as_bmi(Flexure)
    >>> flexure = BmiFlexure()

    >>> config = \"\"\"
    ... eet: 10.e+3
    ... method: flexure
    ... clock:
    ...     start: 0.
    ...     stop: 10.
    ...     step: 2.
    ... grid:
    ...     type: raster
    ...     shape: [20, 40]
    ...     spacing: [1000., 2000.]
    ... \"\"\"
    >>> flexure.initialize(config)
    >>> flexure.get_output_var_names()
    ('lithosphere_surface__elevation_increment',)
    >>> flexure.get_var_grid('lithosphere_surface__elevation_increment')
    0
    >>> flexure.get_grid_shape(0)
    (20, 40)
    >>> dz = flexure.get_value('lithosphere_surface__elevation_increment')
    >>> dz.shape == (800, )
    True

    >>> np.all(dz == 0.)
    True
    >>> flexure.get_current_time()
    0.0

    >>> flexure.get_input_var_names()
    ('lithosphere__overlying_pressure_increment',)
    >>> load = np.zeros((20, 40), dtype=float)
    >>> load[0, 0] = 1.
    >>> flexure.set_value('lithosphere__overlying_pressure_increment', load)
    >>> flexure.update()
    >>> flexure.get_current_time()
    2.0
    >>> dz = flexure.get_value('lithosphere_surface__elevation_increment')
    >>> np.all(dz == 0.)
    False
    """
    if not issubclass(cls, Component):
        raise TypeError("class must inherit from Component")


    class BmiWrapper(object):
        __doc__ = """
        Basic Modeling Interface for the {name} component.
        """.format(
            name=cls.__name__
        ).strip()

        _cls = cls

        def __init__(self):
            self._base = None
            super(BmiWrapper, self).__init__()

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

            BMI-wrapped Landlab components use input files in YAML format.
            Component-specific parameters are listed at the top level,
            followed by grid and then time information. An example input
            file looks like::

                eet: 15.e+3
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

    BmiWrapper.__name__ = cls.__name__
    return BmiWrapper
