#! /usr/bin/env python
"""Interface that describes uniform rectilinear grids."""


class BmiGrid(object):

    """Methods that describe a grid.

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
        """
        pass

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
        """
        pass

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
        """
        pass
