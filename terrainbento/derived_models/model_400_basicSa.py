#! /usr/env/python
"""
model_400_basicSa.py: erosion model using depth-dependent linear
diffusion, basic stream power, and discharge proportional to drainage area.

Model 400 BasicSa

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         FastscapeStreamPower, DepthDependentDiffuser,
                         ExponentialWeatherer


"""

import sys
import numpy as np

from landlab.components import (
    FastscapeEroder,
    DepthDependentDiffuser,
    ExponentialWeatherer,
)
from terrainbento.base_class import ErosionModel


class BasicSa(ErosionModel):
    """
    A BasicSa computes erosion using linear diffusion, basic
    stream power, and Q~A.

    It creates soil through weathering and consideres soil thickness
    in calculating hillslope diffusion.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicSa."""

        # Call ErosionModel's init
        super(BasicSa, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        self.K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor) * self.params[
                "initial_soil_thickness"
            ]  # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor) * self.params[
            "soil_transport_decay_depth"
        ]  # has units length
        max_soil_production_rate = (self._length_factor) * self.params[
            "max_soil_production_rate"
        ]  # has units length per time
        soil_production_decay_depth = (self._length_factor) * self.params[
            "soil_production_decay_depth"
        ]  # has units length

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K_sp,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Create soil thickness (a.k.a. depth) field
        if "soil__depth" in self.grid.at_node:
            soil_thickness = self.grid.at_node["soil__depth"]
        else:
            soil_thickness = self.grid.add_zeros("node", "soil__depth")

        # Create bedrock elevation field
        if "bedrock__elevation" in self.grid.at_node:
            bedrock_elev = self.grid.at_node["bedrock__elevation"]
        else:
            bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")

        # Set soil thickness and bedrock elevation
        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate diffusion and weathering components
        self.diffuser = DepthDependentDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            soil_transport_decay_depth=soil_transport_decay_depth,
        )

        self.weatherer = ExponentialWeatherer(
            self.grid,
            max_soil_production_rate=max_soil_production_rate,
            soil_production_decay_depth=soil_production_decay_depth,
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

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K_sp
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodibility_adjustment_factor()
            )
        self.eroder.run_one_step(dt, flooded_nodes=flooded)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

        # Generate and move soil around
        self.diffuser.run_one_step(dt)

        # Finalize the run_one_step_method
        self.finalize__run_one_step(dt)


def main(): #pragma: no cover
    """Executes model."""
    import sys

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Must include input file name on command line")
        sys.exit(1)

    ldsp = BasicSa(input_file=infile)
    ldsp.run()


if __name__ == "__main__":
    main()
