# coding: utf8
#! /usr/env/python
"""**GenericFuncBaselevelHandler** modifies elevation for all not-core nodes."""


class GenericFuncBaselevelHandler(object):
    """Control the elevation of all nodes that are not core nodes.

    The **GenericFuncBaselevelHandler** controls the elevation of all nodes on
    the model grid with ``status != 0`` (core nodes). The elevation change is
    defined by a generic function of the x and y position across the grid and
    the model time, t. Thus a user is able to use this single BaselevelHandler
    object to make many different uplift patterns, including uplift patterns
    that change as a function of model time.

    Through the parameter ``modify_core_nodes`` the user can determine if the
    core nodes should be moved in the direction (up or down) specified by the
    elevation change directive, or if the non-core nodes should be moved in
    the opposite direction. Negative values returned by the function indicate
    that the core nodes would be uplifted and the not-core nodes would be
    down-dropped.

    The **GenericFuncBaselevelHandler** expects that ``topographic__elevation``
    is an at-node model grid field. It will modify this field as well as
    the field ``bedrock__elevation``, if it exists.

    Note that **GenericFuncBaselevelHandler** increments time at the end of the
    **run_one_step** method.
    """

    def __init__(
        self,
        grid,
        modify_core_nodes=False,
        function=lambda x, y, t: (0 * x + 0 * y + 0 * t),
        **kwargs
    ):
        """
        Parameters
        ----------
        grid : landlab model grid
        modify_core_nodes : boolean, optional
            Flag to indicate if the core nodes or the non-core nodes will
            be modified. Default is False, indicating that nodes in the core
            will be modified.
        function : function, optional
            Function of model grid node x position, y position and model time
            that defines the rate of node elevation change. This function must
            be a function of three variables and return an array of size
            number of nodes. If a constant value is desired, used
            **NotCoreNodeBaselevelHandler** instead. The default function is:
            ``lambda x, y, t: (0*x + 0*y + 0*t)``

        Examples
        --------
        Start by creating a landlab model grid and set its boundary conditions.

        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid(5, 5)
        >>> z = mg.add_zeros('node', 'topographic__elevation')
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
        ...                                        left_is_closed=True,
        ...                                        right_is_closed=True,
        ...                                        top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(
        ...     0, mg.at_node['topographic__elevation'], -9999.)
        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        Now import the **GenericFuncBaselevelHandler** and instantiate.

        >>> from terrainbento.boundary_condition_handlers import (
        ...                                         GenericFuncBaselevelHandler)
        >>> bh = GenericFuncBaselevelHandler(mg,
        ...                                 modify_core_nodes = False,
        ...                                 function = lambda x, y, t: -(x + y + (0*t)))
        >>> bh.run_one_step(10.0)

        We should expect that the boundary nodes (except for node 0) will all
        have lowered by ``10*(x+y)`` in which ``x`` and ``y`` are the node x and
        y positions. The function we provided has no time dependence.

        >>> print(z.reshape(mg.shape))
        [[  0. -10. -20. -30. -40.]
         [-10.   0.   0.   0. -50.]
         [-20.   0.   0.   0. -60.]
         [-30.   0.   0.   0. -70.]
         [-40. -50. -60. -70. -80.]]

        If we wanted instead for all of the non core nodes to change their
        elevation, we would set ``modify_core_nodes = True``. Next we will do
        an example with this option, that also includes a bedrock elevation
        field.

        >>> mg = RasterModelGrid(5, 5)
        >>> z = mg.add_zeros('node', 'topographic__elevation')
        >>> b = mg.add_zeros('node', 'bedrock__elevation')
        >>> b -= 10.
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
        ...                                        left_is_closed=True,
        ...                                        right_is_closed=True,
        ...                                        top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(
        ...     0, mg.at_node['topographic__elevation'], -9999.)
        >>> bh = GenericFuncBaselevelHandler(mg,
        ...                                 modify_core_nodes = True,
        ...                                 function = lambda x, y, t: -(x + y + (0*t)))
        >>> bh.run_one_step(10.0)
        >>> print(z.reshape(mg.shape))
        [[  0.   0.   0.   0.   0.]
         [  0.  20.  30.  40.   0.]
         [  0.  30.  40.  50.   0.]
         [  0.  40.  50.  60.   0.]
         [  0.   0.   0.   0.   0.]]
        >>> print(b.reshape(mg.shape))
        [[-10. -10. -10. -10. -10.]
         [-10.  10.  20.  30. -10.]
         [-10.  20.  30.  40. -10.]
         [-10.  30.  40.  50. -10.]
         [-10. -10. -10. -10. -10.]]

        There is no limit to how complex a function a user can provide. The
        function must only take the variables ``x``, ``y``, and ``t`` and
        return an array of size number of nodes.

        """
        self.model_time = 0.0
        self._grid = grid

        # test the function behaves well
        function_args = function.__code__.co_varnames
        if len(function_args) != 3:
            msg = "GenericFuncBaselevelHandler: function must take only three arguments, x, y, and t."
            raise ValueError(msg)

        test_dzdt = function(
            self._grid.x_of_node, self._grid.y_of_node, self.model_time
        )

        if test_dzdt.shape != self._grid.x_of_node.shape:
            msg = "GenericFuncBaselevelHandler: function must return an array of shape (n_nodes,)"
            raise ValueError(msg)

        self.function = function
        self.modify_core_nodes = modify_core_nodes
        self.z = self._grid.at_node["topographic__elevation"]

        # determine which nodes to lower
        # based on which are lowering, set the prefactor correctly.
        if self.modify_core_nodes:
            self.nodes_to_lower = self._grid.status_at_node == 0
            self.prefactor = -1.0
        else:
            self.nodes_to_lower = self._grid.status_at_node != 0
            self.prefactor = 1.0

    def run_one_step(self, dt):
        """ Run **GenericFuncBaselevelHandler** forward and update elevations.

        The **run_one_step** method provides a consistent interface to update
        the terrainbento boundary condition handlers.

        In the **run_one_step** routine, the **GenericFuncBaselevelHandler** will
        either lower the closed or raise the non-closed nodes based on inputs
        specified at instantiation.

        Note that **GenericFuncBaselevelHandler** increments time at the end of
        the **run_one_step** method.

        Parameters
        ----------
        dt : float
            Duration of model time to advance forward.

        """
        self.dzdt = self.function(
            self._grid.x_of_node, self._grid.y_of_node, self.model_time
        )

        # calculate lowering amount and subtract
        self.z[self.nodes_to_lower] += (
            self.prefactor * self.dzdt[self.nodes_to_lower] * dt
        )

        # if bedrock__elevation exists as a field, lower it also
        if "bedrock__elevation" in self._grid.at_node:
            self._grid.at_node["bedrock__elevation"][self.nodes_to_lower] += (
                self.prefactor * self.dzdt[self.nodes_to_lower] * dt
            )

        # increment model time
        self.model_time += dt
