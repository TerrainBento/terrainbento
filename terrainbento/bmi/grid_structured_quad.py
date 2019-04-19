#! /usr/bin/env python
"""Interface that describes structured quadrilateral grids."""

from .grid import BmiGrid


class BmiGridStructuredQuad(BmiGrid):

    """Methods that describe a structured grid of quadrilaterals.

    .. figure:: _static/grid_structured_quad.png
        :scale: 10%
        :align: center
        :alt: An example of a structured quad grid.
    """

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
        """
        pass
