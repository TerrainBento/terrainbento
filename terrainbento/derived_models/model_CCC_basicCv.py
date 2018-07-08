#! /usr/env/python
"""
model_000_basic.py: erosion model using linear diffusion, basic stream
power, and discharge proportional to drainage area.

Model 000 Basic

Landlab components used: FastscapeStreamPower, LinearDiffuser
"""
import sys
import numpy as np
from scipy.interpolate import interp1d

from landlab.components import FastscapeEroder, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicCv(ErosionModel):
    """
    A BasicCV computes erosion using linear diffusion, basic stream
    power, and Q~A.

    It also has basic climate change
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicCv model."""
        # Call ErosionModel's init
        super(BasicCv, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent("regolith_transport_parameter")

        self.climate_factor = self.params["climate_factor"]
        self.climate_constant_date = self.params["climate_constant_date"]

        time = [0, self.climate_constant_date, self.params["run_duration"]]
        K = [K_sp * self.climate_factor, K_sp, K_sp]
        self.K_through_time = interp1d(time, K)

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid, K_sp=K[0], m_sp=self.params["m_sp"], n_sp=self.params["n_sp"]
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """
        # Route flow
        self.flow_accumulator.run_one_step()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Update erosion based on climate
        self.eroder.K = float(self.K_through_time(self.model_time))

        # Do some erosion (but not on the flooded nodes)
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # Do some soil creep
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main():  # pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    ldsp = BasicCv(input_file=infile)
    ldsp.run()


if __name__ == "__main__":
    main()
