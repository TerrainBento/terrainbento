#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
capture_node_baselevel_handler.py: implements "external" stream capture in an
EMS model by taking control of a specified node, turning it into an open
boundary, and driving its elevation.

Created on Wed Nov 15 10:36:07 2017

@author: gtucker
"""

from landlab import FIXED_VALUE_BOUNDARY


class CaptureNodeBaselevelHandler():
    """CaptureNodeBaselevelHandler turns a given node into an open boundary and
    drives its elevation."""

    def __init__(self, grid, params):

        self.grid = grid
        self.z = grid.at_node['topographic__elevation']
        self.node = params['capture_node']
        self.start = params['capture_start_time']
        try:
            self.stop = params['capture_stabilize_time']
        except KeyError:
            self.stop = params['run_duration']

        try:
            self.post_stabilization_incision_rate = params['post_stabilization_incision_rate']
        except KeyError:
            self.post_stabilization_incision_rate = 0

        self.rate = params['capture_incision_rate']
        self.current_time = 0.0
        self.grid.status_at_node[self.node] = FIXED_VALUE_BOUNDARY

    def run_one_step(self, dt):

        if self.current_time >= self.start and self.current_time < self.stop:
            self.z[self.node] -= self.rate * dt
            print('Lowered cap node by ' + str(self.rate*dt) + ' to ' + str(self.z[self.node]))
        elif self.current_time >= self.stop:
            self.z[self.node] -= self.post_stabilization_incision_rate * dt
            print('Lowered cap node by ' + str(self.post_stabilization_incision_rate*dt) + ' to ' + str(self.z[self.node]))

        self.current_time += dt
