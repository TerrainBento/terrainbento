"""
"""

class UniformPrecipitator(object):
    """
    """

    def __init__(self, mg, rainfall__flux=1.0):
        """
        """
        mg.at_node['rainfall__flux'][:] = rainfall__flux

    def run_one_step(self, dt):
        """
        """
        pass
