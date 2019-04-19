#! /usr/bin/env python
"""Interface that describes unstructured grids."""

from .grid import BmiGrid


class BmiGridUnstructured(BmiGrid):

    """Methods that describe an unstructured grid.

    .. figure:: _static/grid_unstructured.png
        :scale: 10%
        :align: center
        :alt: An example of an unstructured grid.
    """

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

    def get_grid_connectivity(self, grid_id):
        """Get connectivity array of the grid.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like or int
          The graph of connections between the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_connectivity(void * self, int grid_id,
                                      int * connectivity);
        """
        pass

    def get_grid_offset(self, grid_id):
        """Get offsets for the grid nodes.

        Parameters
        ----------
        grid_id : int
          A grid identifier.

        Returns
        -------
        array_like of int
          The offsets for the grid nodes.

        See Also
        --------
        bmi.vars.BmiVars.get_var_grid : Obtain a `grid_id`.

        Notes
        -----
        .. code-block:: c

            /* C */
            int get_grid_offset(void * self, int grid_id, int * offset);
        """
        pass
