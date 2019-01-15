""""""

import numpy as np


class VariableSourceAreaRunoff(object):
    """"""

    def __init__(self, grid, hydraulic_conductivity=0.2):
        """"""

        self._grid = grid

        self._p = grid.at_node["rainfall__flux"]
        self._r = grid.at_node["water__unit_flux_in"]
        self._area = grid.at_node["drainage_area"]
        self._slope = grid.at_node["topographic_steepest_slope"]

        self._H = grid.at_node["soil__depth"]

        self._hydraulic_conductivity = hydraulic_conductivity

    def run_one_step(self, step):
        """"""
        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self._tlam = self._hydraulic_conductivity * self._H * self._grid.dx
        if np.any(self._tlam) <= 0.0:
            raise ValueError(
                "VariableSourceAreaRunoff: Transmissivity must be > 0"
            )

        a = self._tlam * self._slope / self._p

        self._r[:] = (a / self._A ** 2.) * np.exp(-a / self._A)
