"""
"""


class SimpleRunoff(object):
    """
    """

    def __init__(self, mg, runoff_proportion=1.0):
        """
        """
        mg.at_node["water__unit_flux_in"] = (
            runoff_proportion * mg.at_node["rainfall__flux"]
        )

    def run_one_step(self, dt):
        """
        """
        pass
