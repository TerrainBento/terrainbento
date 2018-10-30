"""
"""

class VariableSourceAreaRunoff(object):
    """
    """

    def __init__(self, mg, ):
        """
        """

        self.grid = mg

        self.p =  mg.at_node['rainfall__flux']
        self.r = mg.at_node['water__unit_flux_in']

        self.area = mg.
        self.slope = mg.at_node['topographic_steepest_slope']

        self.soil__initial_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length

        self.K_hydraulic_conductivity = (self._length_factor) * self.params[
            "hydraulic_conductivity"
        ]  # has units length per time

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self.trans = self.K_hydraulic_conductivity * self.soil__initial_thickness

        if self.trans <= 0.0:
            raise ValueError("BasicStVs: Transmissivity must be > 0")

        self.tlam = self.trans * self.grid._dx  # assumes raster

    def run_one_step(self, dt):
        """
        """
        pass

        # Here"s the total (surface + subsurface) discharge
        pa = self.rain_rate * self.area

        # Transmissivity x lambda x slope = subsurface discharge capacity
        tls = self.tlam * self.slope[np.where(self.slope > 0.0)[0]]

        # Subsurface discharge: zero where slope is flat
        self.qss[np.where(self.slope <= 0.0)[0]] = 0.0
        self.qss[np.where(self.slope > 0.0)[0]] = tls * (
            1.0 - np.exp(-pa[np.where(self.slope > 0.0)[0]] / tls)
        )
