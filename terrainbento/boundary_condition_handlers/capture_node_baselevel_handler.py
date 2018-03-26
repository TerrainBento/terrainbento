#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
CaptureNodeBaselevelHandler implements "external" stream capture.
"""

from landlab import FIXED_VALUE_BOUNDARY


class CaptureNodeBaselevelHandler():
    """CaptureNodeBaselevelHandler turns a given node into an open boundary and
    drives its elevation.


    """

    def __init__(self,
                 grid,
                 capture_node = None,
                 capture_start_time = None,
                 capture_stabilize_time = None,
                 run_duration = None,
                 capture_incision_rate = None,
                 post_stabilization_incision_rate = None,
                 **kwargs):

        self.grid = grid
        self.z = grid.at_node['topographic__elevation']
        self.node = capture_node
        self.start = capture_start_time
        try:
            self.stop = capture_stabilize_time
        except KeyError:
            self.stop = run_duration

        try:
            self.post_stabilization_incision_rate = post_stabilization_incision_rate
        except KeyError:
            self.post_stabilization_incision_rate = 0

        self.rate = capture_incision_rate
        self.current_time = 0.0
        self.grid.status_at_node[self.node] = FIXED_VALUE_BOUNDARY

    def run_one_step(self, dt):
        """

        """
        if self.current_time >= self.start and self.current_time < self.stop:
            self.z[self.node] -= self.rate * dt
            print('Lowered cap node by ' + str(self.rate*dt) + ' to ' + str(self.z[self.node]))
        elif self.current_time >= self.stop:
            self.z[self.node] -= self.post_stabilization_incision_rate * dt
            print('Lowered cap node by ' + str(self.post_stabilization_incision_rate*dt) + ' to ' + str(self.z[self.node]))

        self.current_time += dt
