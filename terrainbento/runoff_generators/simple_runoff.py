"""terrainbento **SimpleRunoff**."""


class SimpleRunoff(object):
    """Generate runoff proportional to precipitation.

    **SimpleRunoff** populates the field "water__unit_flux_in" with a value
    proportional to the "rainfall__flux".

    The "water__unit_flux_in" field is accumulated to create discharge.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> from landlab import RasterModelGrid
    >>> from terrainbento import RandomPrecipitator, SimpleRunoff
    >>> grid = RasterModelGrid((5,5))
    >>> precipitator = RandomPrecipitator(grid)
    >>> runoff_generator = SimpleRunoff(grid, runoff_proportion=0.3)
    >>> np.round(
    ...     grid.at_node["rainfall__flux"].reshape(grid.shape),
    ...     decimals=2)
    array([[ 0.37,  0.95,  0.73,  0.6 ,  0.16],
           [ 0.16,  0.06,  0.87,  0.6 ,  0.71],
           [ 0.02,  0.97,  0.83,  0.21,  0.18],
           [ 0.18,  0.3 ,  0.52,  0.43,  0.29],
           [ 0.61,  0.14,  0.29,  0.37,  0.46]])
    >>> np.round(
    ...     grid.at_node["water__unit_flux_in"].reshape(grid.shape),
    ...     decimals=2)
    array([[ 0.11,  0.29,  0.22,  0.18,  0.05],
           [ 0.05,  0.02,  0.26,  0.18,  0.21],
           [ 0.01,  0.29,  0.25,  0.06,  0.05],
           [ 0.06,  0.09,  0.16,  0.13,  0.09],
           [ 0.18,  0.04,  0.09,  0.11,  0.14]])
    """

    def __init__(self, grid, runoff_proportion=1.0):
        """
        Parameters
        ----------
        grid : model grid
        runoff_proportion : float, optional.
            Proportion of "rainfall__flux" that is converted into runoff.
        """
        self._grid = grid
        if "water__unit_flux_in" not in grid.at_node:
            grid.add_ones("node", "water__unit_flux_in")
        self.runoff_proportion = runoff_proportion
        self.run_one_step(0)

    def run_one_step(self, step):
        """Run **SimpleRunoff** forward by duration ``step``"""
        self._grid.at_node["water__unit_flux_in"] = (
            self.runoff_proportion * self._grid.at_node["rainfall__flux"]
        )
