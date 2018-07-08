#! /usr/env/python
"""
model_210_basicHyVs.py: erosion model using linear diffusion,
hybrid alluvium, and discharge proportional to effective drainage
area.

Model 210 BasicHyVs

"vsa" stands for "variable source area".

Landlab components used: FlowRouter, DepressionFinderAndRouter,
                         StreamPowerEroder, LinearDiffuser

"""

import sys
import numpy as np

from landlab.components import ErosionDeposition, LinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicHyVs(ErosionModel):
    """
    A BasicHyVs computes erosion using linear diffusion,
    hybrid alluvium fluvial erosion, and Q ~ A exp( -b S / A).

    "VSA" stands for "variable source area".
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicHyVs."""

        # Call ErosionModel's init
        super(BasicHyVs, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        self.K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        recharge_rate = self._length_factor * self.params["recharge_rate"]  # L/T
        soil_thickness = (
            self._length_factor * self.params["initial_soil_thickness"]
        )  # L
        K_hydraulic_conductivity = (
            self._length_factor * self.params["K_hydraulic_conductivity"]
        )  # has units length per time

        v_sc = self.get_parameter_from_exponent(
            "v_sc"
        )  # normalized settling velocity. Unitless.

        # set methods and fields. K's and sp_crits need to be field names
        method = "simple_stream_power"
        discharge_method = "drainage_area"
        area_field = "effective_drainage_area"
        discharge_field = None

        # Add a field for effective drainage area
        if "effective_drainage_area" in self.grid.at_node:
            self.eff_area = self.grid.at_node["effective_drainage_area"]
        else:
            self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-area parameter
        self.sat_param = (
            K_hydraulic_conductivity * soil_thickness * self.grid.dx
        ) / recharge_rate

        # Handle solver option
        try:
            solver = self.params["solver"]
        except KeyError:
            solver = "original"

        # Instantiate a SPACE component
        self.eroder = ErosionDeposition(
            self.grid,
            K=self.K_sp,
            F_f=self.params["F_f"],
            phi=self.params["phi"],
            v_s=v_sc,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
            method=method,
            discharge_method=discharge_method,
            area_field=area_field,
            discharge_field=discharge_field,
            solver=solver,
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = LinearDiffuser(
            self.grid, linear_diffusivity=regolith_transport_parameter
        )

    def calc_effective_drainage_area(self):
        """Calculate and store effective drainage area.

        Effective drainage area is defined as:

        $A_{eff} = A \exp ( \alpha S / A) = A R_r$

        where $S$ is downslope-positive steepest gradient, $A$ is drainage
        area, $R_r$ is the runoff ratio, and $\alpha$ is the saturation
        parameter.
        """

        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        cores = self.grid.core_nodes
        self.eff_area[cores] = area[cores] * (
            np.exp(-self.sat_param * slope[cores] / area[cores])
        )

    def run_one_step(self, dt):
        """
        Advance model for one time-step of duration dt.
        """

        # Direct and accumulate flow
        self.flow_accumulator.run_one_step()

        # Update effective runoff ratio
        self.calc_effective_drainage_area()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Zero out effective area in flooded nodes
        self.eff_area[flooded] = 0.0

        # Do some erosion
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K_sp
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(dt)

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

    my_model = BasicHyVs(input_file=infile)
    my_model.run()


if __name__ == "__main__":
    main()
