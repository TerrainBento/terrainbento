# coding: utf8
#! /usr/env/python

import sys
import numpy as np

from landlab.io import read_esri_ascii
from terrainbento.base_class import ErosionModel


class TwoLithologyErosionModel(ErosionModel):
    "" ""

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """ """
        # Call ErosionModel's init
        super(TwoLithologyErosionModel, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )
        self.contact_width = (self._length_factor) * self.params[
            "contact_zone__width"
        ]  # has units length

        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]

    def _setup_contact_elevation(self):
        """ """
        file_name = self.params["lithology_contact_elevation__file_name"]

        # Read input data on rock-till contact elevation
        read_esri_ascii(
            file_name, grid=self.grid, halo=1, name="rock_till_contact__elevation"
        )
        self.rock_till_contact = self.grid.at_node["rock_till_contact__elevation"]
