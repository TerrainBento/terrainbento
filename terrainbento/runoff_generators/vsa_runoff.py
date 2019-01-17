"""terrainbento **VariableSourceAreaRunoff**."""

import numpy as np


class VariableSourceAreaRunoff(object):
    """Generate variable source area runoff.

    **VariableSourceAreaRunoff** populates the field "water__unit_flux_in"
    with a value
    proportional to the "rainfall__flux".

    The "water__unit_flux_in" field is accumulated to create discharge.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from terrainbento import UniformPrecipitator, VariableSourceAreaRunoff
    >>> grid = RasterModelGrid((5,5))
    >>> precipitator = UniformPrecipitator(grid)
    >>> runoff_generator = VariableSourceAreaRunoff(grid)
    >>> grid.at_node["rainfall__flux"].reshape(grid.shape)
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    """

    def __init__(self, grid, hydraulic_conductivity=0.2):
        """
        Parameters
        ----------
        grid : model grid
        hydraulic_conductivity : float, optional.
            Hydraulic conductivity. Default is 0.2.
        """
        self._grid = grid

        if "water__unit_flux_in" not in grid.at_node:
            grid.add_ones("node", "water__unit_flux_in")

        self._hydraulic_conductivity = hydraulic_conductivity

    def run_one_step(self, step):
        """Run **VariableSourceAreaRunoff** forward by duration ``step``"""
        self._p = grid.at_node["rainfall__flux"]
        self._r = grid.at_node["water__unit_flux_in"]
        self._area = grid.at_node["drainage_area"]
        self._slope = grid.at_node["topographic_steepest_slope"]
        self._H = grid.at_node["soil__depth"]

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self._tlam = self._hydraulic_conductivity * self._H * self._grid.dx
        if np.any(self._tlam) <= 0.0:
            raise ValueError(
                "VariableSourceAreaRunoff: Transmissivity must be > 0"
            )

        a = self._tlam * self._slope / self._p

        self._r[:] = self._p * (a / self._A ** 2.) * np.exp(-a / self._A)
