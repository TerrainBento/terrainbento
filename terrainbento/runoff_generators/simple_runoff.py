""""""


class SimpleRunoff(object):
    """"""

    def __init__(self, grid, runoff_proportion=1.0):
        """"""
        self._grid = grid
        self.runoff_proportion = runoff_proportion

    def run_one_step(self, step):
        """"""
        self._grid.at_node["water__unit_flux_in"] = (
            self.runoff_proportion * self._grid.at_node["rainfall__flux"]
        )
