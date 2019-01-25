"""terrainbento **UniformPrecipitator**."""


class UniformPrecipitator(object):
    """Generate uniform precipitation.

    UniformPrecipitator populates the at-node field "rainfall__flux" with a
    value provided by the keyword argument ``rainfall_flux``.

    To make discharge proprortional to drainage area, use the default value
    of 1.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from terrainbento import UniformPrecipitator
    >>> grid = RasterModelGrid((5,5))
    >>> precipitator = UniformPrecipitator(grid, rainfall_flux=3.4)
    >>> grid.at_node["rainfall__flux"].reshape(grid.shape)
    array([[ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4]])
    >>> precipitator.run_one_step(10)
    >>> grid.at_node["rainfall__flux"].reshape(grid.shape)
    array([[ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4],
           [ 3.4,  3.4,  3.4,  3.4,  3.4]])
    """

    def __init__(self, grid, rainfall_flux=1.0):
        """
        Parameters
        ----------
        grid : model grid
        rainfall_flux : float, optional
            Rainfall flux. Default value is 1.0.
        """
        self._rainfall_flux = rainfall_flux
        if "rainfall__flux" not in grid.at_node:
            grid.add_ones("node", "rainfall__flux")
        grid.at_node["rainfall__flux"][:] = rainfall_flux

    def run_one_step(self, step):
        """Run **UniformPrecipitator** forward by duration ``step``"""
        pass
