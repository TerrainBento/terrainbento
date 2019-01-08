""""""


class VariableSourceAreaRunoff(object):
    """"""

    def __init__(self, mg):
        """"""

        self.grid = mg

        self.p = mg.at_node["rainfall__flux"]
        self.r = mg.at_node["water__unit_flux_in"]

        self.area = mg.at_node["drainage_area"]
        self.slope = mg.at_node["topographic_steepest_slope"]

        self.H = mg.at_node["soil__depth"]

        self.hydraulic_conductivity = hydraulic_conductivity

    def run_one_step(self, step):
        """"""
        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.tlam = self.hydraulic_conductivity * self.H * self.grid._dx
        if np.any(self.tlam) <= 0.0:
            raise ValueError("VSA Runoff: Transmissivity must be > 0")

        a = self.tlam * self.slope / self.p

        r = (a / self.A ** 2.) * np.exp(-a / self.A)