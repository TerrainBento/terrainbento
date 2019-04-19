#! /usr/bin/env python
"""Interface that describes rectilinear grids."""

from .grid import BmiGrid


class BmiGridRectilinear(BmiGrid):

    """Methods that describe a rectilinear grid.

    In a 2D rectilinear grid, every grid cell (or element) is a rectangle but
    different cells can have different dimensions. All cells in the same row
    have the same grid spacing in the y direction and all cells in the same
    column have the same grid spacing in the x direction. Grid spacings can
    be computed as the difference of successive x or y values.

    .. figure:: _static/grid_rectilinear.png
        :scale: 10%
        :align: center
        :alt: An example of a rectilinear grid
    """

    def get_grid_shape(self, grid_id):
        """Get dimensions of the computational grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        tuple of int
          The dimensions of the grid.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_shape(void * self, const char * var_name,
                               int * shape);
        """
        pass

    def get_grid_x(self, grid_id):
        """Get coordinates of grid nodes in the streamwise direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like of float
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_x(void * self, const char * var_name, double * x);
        """
        pass

    def get_grid_y(self, grid_id):
        """Get coordinates of grid nodes in the transverse direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like of float
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_y(void * self, const char * var_name, double * y);
        """
        pass

    def get_grid_z(self, grid_id):
        """Get coordinates of grid nodes in the normal direction.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like of float
          The positions of the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_z(void * self, const char * var_name, double * z);
        """
        pass
