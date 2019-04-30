"""terrainbento **RandomPrecipitator**."""

import numpy as np


class RandomPrecipitator(object):
    """Generate random precipitation.

    **RandomPrecipitator** populates the at-node field "rainfall__flux" with
    random values drawn from a distribution. All distributions provided in the
    `numpy.random submodule
    <https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.random.html#distributions>`_
    are supported.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> from landlab import RasterModelGrid
    >>> from terrainbento import RandomPrecipitator
    >>> grid = RasterModelGrid((5,5))
    >>> precipitator = RandomPrecipitator(grid)
    >>> np.round(
    ...     grid.at_node["rainfall__flux"].reshape(grid.shape),
    ...     decimals=2)
    array([[ 0.37,  0.95,  0.73,  0.6 ,  0.16],
           [ 0.16,  0.06,  0.87,  0.6 ,  0.71],
           [ 0.02,  0.97,  0.83,  0.21,  0.18],
           [ 0.18,  0.3 ,  0.52,  0.43,  0.29],
           [ 0.61,  0.14,  0.29,  0.37,  0.46]])
    >>> precipitator.run_one_step(10)
    >>> np.round(
    ...     grid.at_node["rainfall__flux"].reshape(grid.shape),
    ...     decimals=2)
    array([[ 0.79,  0.2 ,  0.51,  0.59,  0.05],
           [ 0.61,  0.17,  0.07,  0.95,  0.97],
           [ 0.81,  0.3 ,  0.1 ,  0.68,  0.44],
           [ 0.12,  0.5 ,  0.03,  0.91,  0.26],
           [ 0.66,  0.31,  0.52,  0.55,  0.18]])
    """

    def __init__(self, grid, distribution="uniform", **kwargs):
        """
        Parameters
        ----------
        grid : model grid
        distribution : str, optional
            Name of the distribution provided by the np.random
            submodule. Default is "uniform".
        kwargs : dict
            Keyword arguments to pass to the ``np.random`` distribution
            function.
        """
        self._grid = grid
        if "rainfall__flux" not in grid.at_node:
            grid.add_ones("node", "rainfall__flux")
        self.function = np.random.__dict__[distribution]
        self._kwargs = kwargs
        self.run_one_step(0.0)

    def run_one_step(self, step):
        """Run **RandomPrecipitator** forward by duration ``step``"""
        values = self.function(size=self._grid.size("node"), **self._kwargs)
        self._grid.at_node["rainfall__flux"][:] = values
