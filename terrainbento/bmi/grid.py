#! /usr/bin/env python
"""Interface that describes uniform rectilinear grids."""


class BmiGrid(object):

    """Methods that describe a grid.

    Four grid types are supported: unstructured, structured quadrilateral,
    rectilinear, and uniform rectilinear. Each grid type requires a different
    set of BMI functions. Additionally, depending on the grid rank (0, 1, 2, 3)
    cooresponding to scalar, 1D, 2D, or 3D models, the BMI functions may
    change. If a function is not appropriate for a particular model, it should
    raise a `NotImplementedError`.

    terrainbento uses either uniform rectilinear grids or unstructured grids.
    As terrainbento creates 2D models (the third dimension is the  state
    variable, "topographic__elevation"), they are of rank 2.


    Function                Unstructured    Structured quadrilateral    Rectilinear     Uniform rectilinear
    ========                ============    ========================    ===========     ===================
    get_grid_rank           x               x                           x               x
    get_grid_size           x               x                           x               x
    get_grid_type           x               x                           x               x

    get_grid_x              if rank > 0     x                           x
    get_grid_y              if rank > 1     x                           x
    get_grid_z              if rank >  2    x                           x

    get_grid_edge_count     x
    get_grid_face_count     x
    get_grid_node_count     x

    get_grid_edge_nodes     x
    get_grid_face_nodes     x
    get_grid_face_edges     x
    get_grid_nodes_per_face x

    get_grid_shape                          x                          x                x

    get_grid_spacing                                                                    x
    get_grid_origin                                                                     x


    Todo: add links to functions.

    1. Methods that describe an unstructured grid.

    This is the most general grid type and can be used for any type of grid.
    However, most grids that consist of 4-sided polygons can be represented
    using one of the other grid types. This grid type must be used if the grid
    consists of any elements or cells that do not have four sides.

    Note that a grid of equilateral triangles, while is most certainly
    structured, would need to be represented as an unstructured grid. The same
    is true for a grid of hexagons.

    .. figure:: _static/grid_unstructured.png
        :scale: 10%
        :align: center
        :alt: An example of an unstructured grid.

    2. Methods that describe a structured grid of quadrilaterals.

    The most general structured quadrilateral grid is one whose rows (and
    columns) do not share a common coordinate. In this case, coordinates are
    required for each grid element.

    .. figure:: _static/grid_structured_quad.png
        :scale: 10%
        :align: center
        :alt: An example of a structured quad grid.

    3. Methods that describe a rectilinear grid.

    In a 2D rectilinear grid, every grid cell (or element) is a rectangle but
    different cells can have different dimensions. All cells in the same row
    have the same grid spacing in the y direction and all cells in the same
    column have the same grid spacing in the x direction. Grid spacings can
    be computed as the difference of successive x or y values.

    .. figure:: _static/grid_rectilinear.png
        :scale: 10%
        :align: center
        :alt: An example of a rectilinear grid

    4. Methods that describe a uniform rectilinear grid.

    In a 2D uniform grid, every grid cell (or element) is a rectangle and all
    cells have the same dimensions. If the dimensions are equal, then the
    grid is a tiling of squares.

    Each of these functions returns information about each dimension of a
    grid. The dimensions are ordered with "ij" indexing (as opposed to "xy").
    For example, the :func:`get_grid_shape` function for the example grid would
    return the array ``[4, 5]``. If there were a third dimension, the length of
    the z dimension would be listed first. This same convention is used in
    NumPy. Note that the grid shape is the number of nodes in the coordinate
    directions and not the number of cells or elements. It is possible for
    grid values to be associated with the nodes or with the cells.

    .. figure:: _static/grid_uniform_rectilinear.png
        :scale: 10%
        :align: center
        :alt: An example of a uniform rectilinear grid

    """

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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_rank(grid_id)
        """
        # needed for all grid types
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_size()
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
          Type of grid as a string. Valid return values are
          "uniform_rectilinear", "rectilinear", "structured_quadrilateral",
          and "unstructured".

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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_type(grid_id)
        """
        raise NotImplementedError()

    def get_grid_x(self, grid_id):
        """Get coordinates of grid nodes in the streamwise direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_x(void * self, int grid_id, double * x);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_x(grid_id)
        """
        raise NotImplementedError()

    def get_grid_y(self, grid_id):
        """Get coordinates of grid nodes in the transverse direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_y(void * self, int grid_id, double * y);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_y(grid_id)
        """
        raise NotImplementedError()

    def get_grid_z(self, grid_id):
        """Get coordinates of grid nodes in the normal direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_z(void * self, int grid_id, double * z);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_z(grid_id)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_edge_count(grid_id)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_edge_nodes(grid_id, edge_nodes)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_face_count(grid_id)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_face_nodes(grid_id, face_nodes)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_node_count(grid_id)
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
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_nodes_per_face(grid_id, nodes_per_face)
        """
        raise NotImplementedError()

    def get_grid_shape(self, grid_id):
        """Get dimensions of the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The dimensions of the grid.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_shape(void * self, int grid_id, int * shape);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_shape(grid_id)
        """
        raise NotImplementedError()

    def get_grid_spacing(self, grid_id):
        """Get distance between nodes of the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The grid spacing.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_spacing(void * self, int grid_id, double * spacing);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_spacing(grid_id)
        """
        raise NotImplementedError()

    def get_grid_origin(self, grid_id):
        """Get coordinates for the origin of the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like
          The coordinates of the lower left corner of the grid.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_origin(void * self, int grid_id, double * origin);

        Examples
        --------
        >>> from module import object
        >>> instance = Object.initialize(example_input_file)
        >>> instance.get_grid_origin(grid_id)
        """
        raise NotImplementedError()

def get_grid_face_edges(self, grid_id):
    """
    """
    raise NotImplementedError()
