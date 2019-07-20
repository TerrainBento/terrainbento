# coding: utf8
# !/usr/env/python
"""
**CaptureNodeBaselevelHandler** implements "external" stream capture.
"""

from landlab import FIXED_VALUE_BOUNDARY


class CaptureNodeBaselevelHandler(object):
    """Turn a closed boundary node into an open, lowering, boundary node.

    A **CaptureNodeBaselevelHandler** turns a given node into an open boundary
    and lowers its elevation over time. This is meant as a simple approach to
    model stream capture external to the modeled basin.

    Note that **CaptureNodeBaselevelHandler** increments time at the end of the
    **run_one_step** method.
    """

    def __init__(
        self,
        grid,
        capture_node=None,
        capture_start_time=0,
        capture_stop_time=None,
        capture_incision_rate=-0.01,
        post_capture_incision_rate=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        grid : landlab model grid
        capture_node : int
            Node id of the model grid node that should be captured.
        capture_start_time : float, optional
            Time at which capture should begin. Default is at onset of model
            run.
        capture_stop_time : float, optional
            Time at which capture ceases. Default is the entire duration of
            model run.
        capture_incision_rate : float, optional
            Rate of capture node elevation change.  Units are implied by the
            model grids spatial scale and the time units of ``step``. Negative
            values mean the outlet lowers. Default value is -0.01.
        post_capture_incision_rate : float, optional
            Rate of captured node elevation change after capture ceases.  Units
            are implied by the model grids spatial scale and the time units of
            ``step``. Negative values mean the outlet lowers. Default value is 0.

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

        Now import the **CaptureNodeBaselevelHandler** and instantiate.

        >>> from terrainbento.boundary_handlers import (
        ...                                       CaptureNodeBaselevelHandler)
        >>> bh = CaptureNodeBaselevelHandler(mg,
        ...                                  capture_node = 3,
        ...                                  capture_incision_rate = -3.0,
        ...                                  capture_start_time = 10,
        ...                                  capture_stop_time = 20,
        ...                                  post_capture_incision_rate = -0.1)
        >>> for _ in range(10):
        ...     bh.run_one_step(1)

        The capture has not yet started, so we should expect that the
        topography is still all zeros.

        >>> print(z.reshape(mg.shape))
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]

        Running forward another 10 time units, we should
        see node 3 lower by 30.

        >>> for _ in range(10):
        ...     bh.run_one_step(1)
        >>> print(z.reshape(mg.shape))
        [[  0.   0.   0. -30.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]]
        >>> bh.model_time
        20.0

        Now that model time has reached 20, lowering will occur at the post-
        capture incision rate. The node should lower by 1 to -31 in the next
        10 time units.

        >>> for _ in range(10):
        ...     bh.run_one_step(1)
        >>> print(z.reshape(mg.shape))
        [[  0.   0.   0. -31.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.]]

        """
        self.model_time = 0.0
        self._grid = grid
        self.z = grid.at_node["topographic__elevation"]
        self.node = capture_node
        self.start = capture_start_time
        self.rate = capture_incision_rate

        if capture_stop_time is None:
            self.capture_ends = False
        else:
            self.capture_ends = True
            self.stop = capture_stop_time

        if post_capture_incision_rate is None:
            self.post_capture_incision_rate = 0
        else:
            self.post_capture_incision_rate = post_capture_incision_rate

        self._grid.status_at_node[self.node] = FIXED_VALUE_BOUNDARY

    def run_one_step(self, step):
        """Run **CaptureNodeBaselevelHandler** to update captured node
        elevation.

        The **run_one_step** method provides a consistent interface to update
        the terrainbento boundary condition handlers.

        In the **run_one_step** routine, the **CaptureNodeBaselevelHandler**
        will determine if capture is occuring and change the elevation of the
        captured node based on the amount specified in instantiation.

        Note that **CaptureNodeBaselevelHandler** increments time at the end of
        the **run_one_step** method.

        Parameters
        ----------
        step : float
            Duration of model time to advance forward.
        """
        # lower the correct amount.
        if self.model_time >= self.start:
            if self.capture_ends:
                if self.model_time < self.stop:
                    self.z[self.node] += self.rate * step
                else:
                    self.z[self.node] += self.post_capture_incision_rate * step
            else:
                self.z[self.node] += self.rate * step
        # increment model time
        self.model_time += step
