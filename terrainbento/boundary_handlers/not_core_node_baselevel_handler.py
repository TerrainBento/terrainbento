# coding: utf8
# !/usr/env/python
"""**NotCoreNodeBaselevelHandler** modifies elevation for not-core nodes."""

import os

import numpy as np
from scipy.interpolate import interp1d


class NotCoreNodeBaselevelHandler(object):
    """Control the elevation of all nodes that are not core nodes.

    The **NotCoreNodeBaselevelHandler** controls the elevation of all nodes on
    the model grid with ``status != 0`` (i.e., all not-core nodes). The
    elevation change at these nodes is specified either as a constant rate, or
    through a text file that specifies the elevation change through time.

    Through the parameter ``modify_core_nodes`` the user can determine if the
    core nodes should be moved in the direction (up or down) specified by the
    elevation change directive, or if the non-core nodes should be moved in
    the opposite direction.

    The **NotCoreNodeBaselevelHandler** expects that ``topographic__elevation``
    is an at-node model grid field. It will modify this field as well as
    the field ``bedrock__elevation``, if it exists.

    Note that **NotCoreNodeBaselevelHandler** increments time at the end of the
    **run_one_step** method.
    """

    def __init__(
        self,
        grid,
        modify_core_nodes=False,
        lowering_rate=None,
        lowering_file_path=None,
        model_end_elevation=None,
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
        lowering_rate : float, optional
            Lowering rate of the outlet node. One of ``lowering_rate`` and
            ``lowering_file_path`` is required. Units are implied by the
            model grids spatial scale and the time units of ``step``. Negative
            values mean that the outlet lowers.
        lowering_file_path : str, optional
            Lowering history file path. One of ``lowering_rate``
            and `lowering_file_path` is required. Units are implied by
            the model grids spatial scale and the time units of ``step``.
            This file should be readable with
            ``np.loadtxt(filename, skiprows=1, delimiter=",")``
            Its first column is time and its second colum is the elevation
            change at the outlet since the onset of the model run. Negative
            values mean the outlet lowers.
        model_end_elevation : float, optional
            Average elevation of the nodes_to_lower at the end of the model
            run duration. When the outlet is lowered based on an
            lowering_file_path, a ``model_end_elevation`` can be set such that
            lowering is scaled based on the starting and ending outlet
            elevation. Default behavior is to not scale the lowering pattern.

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

        Now import the **NotCoreNodeBaselevelHandler** and instantiate.

        >>> from terrainbento.boundary_handlers import (
        ...                                      NotCoreNodeBaselevelHandler)
        >>> bh = NotCoreNodeBaselevelHandler(mg,
        ...                                 modify_core_nodes = False,
        ...                                 lowering_rate = -0.1)
        >>> bh.run_one_step(10.0)

        We should expect that the boundary nodes (except for node 0) will all
        have lowered by -1.

        >>> print(z.reshape(mg.shape))
        [[-1. -1. -1. -1. -1.]
         [-1.  0.  0.  0. -1.]
         [-1.  0.  0.  0. -1.]
         [-1.  0.  0.  0. -1.]
         [-1. -1. -1. -1. -1.]]

        If we wanted instead for all of the non core nodes to change their
        elevation, we would set ``modify_core_nodes = True``.

        >>> mg = RasterModelGrid((5, 5))
        >>> z = mg.add_zeros("node", "topographic__elevation")
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,
        ...                                        left_is_closed=True,
        ...                                        right_is_closed=True,
        ...                                        top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(
        ...     0, mg.at_node["topographic__elevation"], -9999.)
        >>> from terrainbento.boundary_handlers import (
        ...                                        NotCoreNodeBaselevelHandler)
        >>> bh = NotCoreNodeBaselevelHandler(mg,
        ...                                 modify_core_nodes = True,
        ...                                 lowering_rate = -0.1)
        >>> bh.run_one_step(10.0)
        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  1.  1.  1.  0.]
         [ 0.  1.  1.  1.  0.]
         [ 0.  1.  1.  1.  0.]
         [ 0.  0.  0.  0.  0.]]

        More complex baselevel histories can be provided with a
        ``lowering_file_path``.

        """
        self.model_time = 0.0
        self._grid = grid
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

        if (lowering_file_path is None) and (lowering_rate is None):
            raise ValueError(
                (
                    "NotCoreNodeBaselevelHandler requires one of "
                    "lowering_rate and lowering_file_path"
                )
            )
        else:
            if lowering_rate is None:
                # initialize outlet elevation object
                if os.path.exists(lowering_file_path):

                    elev_change_df = np.loadtxt(
                        lowering_file_path, skiprows=1, delimiter=","
                    )
                    time = elev_change_df[:, 0]
                    elev_change = elev_change_df[:, 1]

                    model_start_elevation = np.mean(
                        self.z[self.nodes_to_lower]
                    )

                    if model_end_elevation is None:
                        self.scaling_factor = 1.0
                    else:
                        self.scaling_factor = np.abs(
                            model_start_elevation - model_end_elevation
                        ) / np.abs(elev_change[0] - elev_change[-1])

                    outlet_elevation = (
                        self.scaling_factor
                        * self.prefactor
                        * elev_change_df[:, 1]
                    ) + model_start_elevation

                    self.outlet_elevation_obj = interp1d(
                        time, outlet_elevation
                    )
                    self.lowering_rate = None
                else:
                    raise ValueError(
                        (
                            "The lowering_file_path provided "
                            "to NotCoreNodeBaselevelHandler does not "
                            "exist."
                        )
                    )
            elif lowering_file_path is None:
                self.lowering_rate = lowering_rate
                self.outlet_elevation_obj = None
            else:
                raise ValueError(
                    (
                        "Both an lowering_rate and a "
                        "lowering_file_path have been provided "
                        "to NotCoreNodeBaselevelHandler. Please provide "
                        "only one."
                    )
                )

    def run_one_step(self, step):
        """Run **NotCoreNodeBaselevelHandler** forward and update elevations.

        The **run_one_step** method provides a consistent interface to update
        the terrainbento boundary condition handlers.

        In the **run_one_step** routine, the **NotCoreNodeBaselevelHandler**
        will either lower the closed or raise the non-closed nodes based on
        inputs specified at instantiation.

        Note that **NotCoreNodeBaselevelHandler** increments time at the end of
        the **run_one_step** method.

        Parameters
        ----------
        step : float
            Duration of model time to advance forward.
        """
        # next, lower the correct nodes the desired amount
        # first, if we do not have an outlet elevation object
        if self.outlet_elevation_obj is None:

            # calculate lowering amount and subtract
            self.z[self.nodes_to_lower] += (
                self.prefactor * self.lowering_rate * step
            )

            # if bedrock__elevation exists as a field, lower it also
            other_fields = [
                "bedrock__elevation",
                "lithology_contact__elevation",
            ]
            for of in other_fields:
                if of in self._grid.at_node:
                    self._grid.at_node[of][self.nodes_to_lower] += (
                        self.prefactor * self.lowering_rate * step
                    )

        # if there is an outlet elevation object
        else:
            # if bedrock__elevation exists as a field, lower it also
            # calcuate the topographic change required to match the current
            # time"s value for outlet elevation. This must be done in case
            # bedrock elevation exists, and must be done before the topography
            # is lowered
            mean_z = np.mean(self.z[self.nodes_to_lower])
            self.topo_change = mean_z - self.outlet_elevation_obj(
                self.model_time
            )

            other_fields = [
                "bedrock__elevation",
                "lithology_contact__elevation",
            ]
            for of in other_fields:
                if of in self._grid.at_node:
                    self._grid.at_node[of][
                        self.nodes_to_lower
                    ] -= self.topo_change

            # lower topography
            self.z[self.nodes_to_lower] -= self.topo_change

        # increment model time
        self.model_time += step
