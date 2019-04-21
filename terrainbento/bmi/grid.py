#! /usr/bin/env python
"""Interface that describes uniform rectilinear grids."""


class BmiGrid(object):

    """Methods that describe a grid."""

    def get_grid_rank(self, grid_id):
        """Get number of dimensions of the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        int
          Rank of the grid.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_rank(void * self, int grid_id, int * rank);

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_size(self, grid_id):
        """Get the total number of elements in the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        int
          Size of the grid.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_size(void * self, int grid_id, int * size);

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_type(self, grid_id):
        """Get the grid type as a string.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        str
          Type of grid as a string.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_type(void * self, int grid_id, char * type);

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_edge_count(self, grid_id):
        """Get the number of edges in the grid.

        Parameters
        ----------
        grid_id : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid edges.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_edge_nodes(self, grid_id, edge_nodes):
        """Get the edge-node connectivity.

        Parameters
        ----------
        grid_id : int
            A grid identifier.
        edge_nodes : ndarray of int, shape *(2 x nnodes,)*
            A numpy array to place the edge-node connectivity. For each edge,
            connectivity is given as node at edge tail, followed by node at
            edge head.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the edge-node connectivity.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_face_count(self, grid_id):
        """Get the number of faces in the grid.

        Parameters
        ----------
        grid_id : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid faces.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_face_nodes(self, grid_id, face_nodes):
        """Get the face-node connectivity.

        Parameters
        ----------
        grid_id : int
            A grid identifier.
        face_nodes : ndarray of int
            A numpy array to place the face-node connectivity. For each face,
            the nodes (listed in a counter-clockwise direction) that form the
            boundary of the face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-node connectivity.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_node_count(self, grid_id):
        """Get the number of nodes in the grid.

        Parameters
        ----------
        grid_id : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid nodes.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()

    def get_grid_nodes_per_face(self, grid_id, nodes_per_face):
        """Get the number of nodes for each face.

        Parameters
        ----------
        grid_id : int
            A grid identifier.
        nodes_per_face : ndarray of int, shape *(nfaces,)*
            A numpy array to place the number of edges per face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the number of nodes per edge.

        Notes
        -----
        .. code-block:: c

            /* C */
            TODO

        Examples
        --------
        >>> # insert model specific example here.
        """
        raise NotImplementedError()
