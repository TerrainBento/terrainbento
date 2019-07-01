# coding: utf8
# !/usr/env/python
"""**GenericFuncBaselevelHandler** modifies elevation for not-core nodes."""


class GenericFuncBaselevelHandler(object):
    """Control the elevation of all nodes that are not core nodes.

    The **GenericFuncBaselevelHandler** controls the elevation of all nodes on
    the model grid with ``status != 0`` (i.e., all not-core nodes). The
    elevation change is defined by a generic function of the x and y position
    across the grid and the model time, t. Thus a user is able to use this
    single BaselevelHandler object to make many different uplift patterns,
    including uplift patterns that change as a function of model time.

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
        function=lambda grid, t: (
            0 * grid.x_of_node + 0 * grid.y_of_node + 0 * t
        ),
        **kwargs
    ):
        """
        Parameters
        ----------
        grid : landlab model grid
        modify_core_nodes : boolean, optional
            Flag to indicate if the core nodes or the non-core nodes will
            be modified. Default is False, indicating that the boundary nodes
            will be modified.
        function : function, optional
            Function of model grid node x position, y position and model time
            that defines the rate of node elevation change. This function must
            be a function of three variables and return an array of size
            number of nodes. If a constant value is desired, used
            **NotCoreNodeBaselevelHandler** instead. The default function is:
            ``lambda grid, t: (0 * grid.x_of_node + 0 * grid.y_of_node + 0 * t)``

        Examples
        --------
        Start by creating a landlab model grid and set its boundary conditions.

        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((5, 5))
        >>> z = mg.add_zeros("node", "topographic__elevation")
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
        ...                                        left_is_closed=True,
        ...                                        right_is_closed=True,
        ...                                        top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(
        ...     0, mg.at_node["topographic__elevation"], -9999.)
        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        Now import the **GenericFuncBaselevelHandler** and instantiate.

        >>> from terrainbento.boundary_handlers import (
        ...                                       GenericFuncBaselevelHandler)
        >>> my_func = lambda grid, t:-(grid.x_of_node + grid.y_of_node + (0*t))
        >>> bh = GenericFuncBaselevelHandler(mg,
        ...                                  modify_core_nodes = False,
        ...                                   function=my_func)
        >>> bh.run_one_step(10.0)

        We should expect that the boundary nodes (except for node 0) will all
        have lowered by ``10*(x+y)`` in which ``x`` and ``y`` are the node x
        and y positions. The function we provided has no time dependence.

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

        >>> mg = RasterModelGrid((5, 5))
        >>> z = mg.add_zeros("node", "topographic__elevation")
        >>> b = mg.add_zeros("node", "bedrock__elevation")
        >>> b -= 10.
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
        ...                                        left_is_closed=True,
        ...                                        right_is_closed=True,
        ...                                        top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(
        ...     0, mg.at_node["topographic__elevation"], -9999.)
        >>> my_func = lambda grid, t: -(grid.x_of_node + grid.y_of_node)
        >>> bh = GenericFuncBaselevelHandler(mg,
        ...                                 modify_core_nodes = True,
        ...                                 function=my_func)
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
        function must only take the variables ``grid``, and ``t`` and
        return an array that represents the desired rate of surface elevation
        change (dzdt) at each node.

        If a user wanted to use this function to implement boundary conditions
        that involved modifying the grid, but not necessarily modifying the
        elevation of core or not-core nodes, then the function could modify the
        grid in the desired way and then return an array of zeros of size
        (n_nodes,).

        """
        self.model_time = 0.0
        self._grid = grid

        # test the function behaves well
        if function.__code__.co_argcount != 2:
            msg = (
                "GenericFuncBaselevelHandler: function must take only two "
                "arguments, grid and t."
            )
            raise ValueError(msg)

        test_dzdt = function(self._grid, self.model_time)

        if hasattr(test_dzdt, "shape"):
            if test_dzdt.shape != self._grid.x_of_node.shape:
                msg = (
                    "GenericFuncBaselevelHandler: function must return an "
                    "array of shape (n_nodes,)"
                )
                raise ValueError(msg)
        else:
            msg = (
                "GenericFuncBaselevelHandler: function must return an "
                "array of shape (n_nodes,)"
            )
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

    def run_one_step(self, step):
        """Run **GenericFuncBaselevelHandler** forward and update elevations.

        The **run_one_step** method provides a consistent interface to update
        the terrainbento boundary condition handlers.

        In the **run_one_step** routine, the **GenericFuncBaselevelHandler**
        will either lower the closed or raise the non-closed nodes based on
        inputs specified at instantiation.

        Note that **GenericFuncBaselevelHandler** increments time at the end of
        the **run_one_step** method.

        Parameters
        ----------
        step : float
            Duration of model time to advance forward.
        """
        self.dzdt = self.function(self._grid, self.model_time)

        # calculate lowering amount and subtract
        self.z[self.nodes_to_lower] += (
            self.prefactor * self.dzdt[self.nodes_to_lower] * step
        )

        # if bedrock__elevation exists as a field, lower it also
        other_fields = ["bedrock__elevation", "lithology_contact__elevation"]
        for of in other_fields:
            if of in self._grid.at_node:
                self._grid.at_node[of][self.nodes_to_lower] += (
                    self.prefactor * self.dzdt[self.nodes_to_lower] * step
                )

        # increment model time
        self.model_time += step
