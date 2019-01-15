""""""


class UniformPrecipitator(object):
    """UniformPrecipitator."""

    def __init__(self, mg, rainfall_flux=1.0):
        """"""
        mg.at_node["rainfall__flux"][:] = rainfall_flux

    def run_one_step(self, step):
        """"""
        pass
