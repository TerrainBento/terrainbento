#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
``NoiseAdder`` adds noise to Landlab grid fields.
"""


class NoiseAdder():
    """Turn a closed boundary node into an open, lowering, boundary node.

    ``CaptureNodeBaselevelHandler`` turns a given node into an open boundary and
    changing elevation. This is meant as a simple approach to model stream
    capture external to the modeled basin.

    Note that ``CaptureNodeBaselevelHandler`` increments time at the end of the
    ``run_one_step`` method.

    Methods
    -------
    run_one_step

    """

    def __init__(self,
                 grid,
                 noise_iterable,
                 **kwargs):
        """
        Parameters
        ----------
        grid : landlab model grid
        noise_iterable : iterable
            Iterable of dictionaries, each of which is associated with a grid
            field to add noise to.
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

        Now import the ``CaptureNodeBaselevelHandler`` and instantiate.

        >>> from terrainbento.boundary_condition_handlers import (
        ...                                         CaptureNodeBaselevelHandler)
        >>> bh = CaptureNodeBaselevelHandler(mg,
        ...                                  capture_node = 3,
        ...                                  capture_incision_rate = -3.0,
        ...                                  capture_start_time = 10,
        ...                                  capture_stop_time = 20,
        ...                                  post_capture_incision_rate = -0.1)
        >>> for i in range(10):
        ...     bh.run_one_step(1)

        The capture has not yet started, so we should expect that the topography
        is still all zeros.

        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        Running forward another 10 time units, we should
        see node three lower by 30.

        >>> for i in range(10):
        ...     bh.run_one_step(1)
        >>> print(z.reshape(mg.shape))
        [[  0.   0.   0. -30.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]]
        >>> bh.model_time
        20.0

        Now that model time has reached 20, lowering will occur at the post
        capture incision rate. The node should lower by 1 to -31 in the next
        10 time units.

        >>> for i in range(10):
        ...     bh.run_one_step(1)
        >>> print(z.reshape(mg.shape))
        [[  0.   0.   0. -31.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]]

        """
        self.model_time = 0.0
        self.grid = grid



    def run_one_step(self, dt):
        """
        Run ``CaptureNodeBaselevelHandler`` forward and update outlet node elevation.

        The ``run_one_step`` method provides a consistent interface to update
        the ``terrainbento`` boundary condition handlers.

        In the ``run_one_step`` routine, the ``CaptureNodeBaselevelHandler``
        will determine if capture is occuring and change the elevation of the
        captured node based on the amount specified in instantiation.

        Note that ``CaptureNodeBaselevelHandler`` increments time at the end of
        the ``run_one_step`` method.

        Parameters
        ----------
        dt : float
            Duration of model time to advance forward.

        """
        # lower the correct amount.

        # increment model time
        self.model_time += dt
