""""""

import numpy as np


class RandomPrecipitator(object):
    """"""

    def __init__(self, mg, distribution, **kwargs):
        """
        Parameters
        ----------
        grid
        distribution : str, optional
            Name of the distribution provided by the np.random
            submodule.
        kwargs : dict
            Keyword arguments to pass to the ``np.random`` distribution
            function.

        """
        self._grid = mg
        if distribution not in np.random.__dict__:
            raise ValueError("")
        self.function = np.random.__dict__[distribution]
        self.run_one_step(0.)

    def run_one_step(self, step):
        """"""
        values = self.function(self._grid.size("node"), **kwargs)
        mg.at_node["rainfall__flux"][:] = values
