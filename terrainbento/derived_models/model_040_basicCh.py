#! /usr/env/python
""" ``terrainbento`` Model ``BasicCh`` program.

Erosion model using cubic diffusion, basic stream
power, and discharge proportional to drainage area.


Landlab components used: 
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`
    4. `TaylorNonLinearDiffuser <http://landlab.readthedocs.io/en/release/landlab.components.taylor_nonlinear_hillslope_flux.html>`

"""

import sys
import numpy as np

from landlab.components import FastscapeEroder, TaylorNonLinearDiffuser
from terrainbento.base_class import ErosionModel


class BasicCh(ErosionModel):
    """
    
    Model ``BasicCh`` is a model program that evolves a topographic surface 
    described by :math:`\eta` with the following governing equation:

    .. math::

        

    A BasicCh computes erosion using cubic diffusion, basic stream
    power, and Q~A.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """Initialize the BasicCh."""

        # Call ErosionModel's init
        super(BasicCh, self).__init__(
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

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K_sp,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a LinearDiffuser component
        self.diffuser = TaylorNonLinearDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=self.params["slope_crit"],
            nterms=11,
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

        # Do some soil creep
        self.diffuser.run_one_step(
            dt, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

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

    cdsp = BasicCh(input_file=infile)
    cdsp.run()


if __name__ == "__main__":
    main()
