# coding: utf8
#! /usr/env/python
"""terrainbento model **BasicSaVs** program.

Erosion model using depth-dependent linear diffusion with a soil layer, basic
stream power, and discharge proportional to effective drainage area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `DepthDependentDiffuser <http://landlab.readthedocs.io/en/release/_modules/landlab/components/depth_dependent_diffusion/hillslope_depth_dependent_linear_flux.html#DepthDependentDiffuser>`_
    5. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/_modules/landlab/components/weathering/exponential_weathering.html#ExponentialWeatherer>`_

"""

import sys
import numpy as np

from landlab.components import (
    StreamPowerEroder,
    DepthDependentDiffuser,
    ExponentialWeatherer,
)
from terrainbento.base_class import ErosionModel


class BasicSaVs(ErosionModel):
    """
    A BasicSaVs computes erosion using depth-dependent linear diffusion, basic
    stream power, and Q ~ A exp( -c H S / A); H = soil thickness.

    This "c" parameter has dimensions of length, and is defined as
    c = K dx / R, where K is saturated hydraulic conductivity, dx is grid
    spacing, and R is recharge.
    """

    def __init__(
        self, input_file=None, params=None, BoundaryHandlers=None, OutputWriters=None
    ):
        """
        Parameters
        ----------
        input_file : str
            Path to model input file. See wiki for discussion of input file
            formatting. One of input_file or params is required.
        params : dict
            Dictionary containing the input file. One of input_file or params is
            required.
        BoundaryHandlers : class or list of classes, optional
            Classes used to handle boundary conditions. Alternatively can be
            passed by input file as string. Valid options described above.
        OutputWriters : class, function, or list of classes and/or functions, optional
            Classes or functions used to write incremental output (e.g. make a
            diagnostic plot).

        Returns
        -------
        BasicSaVs : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model **BasicSaVs**. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the terrainbento tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicSaVs

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'soil__initial_thickness': 0.0,
        ...           'soil_transport_decay_depth': 0.2,
        ...           'soil_production__maximum_rate': 0.001,
        ...           'soil_production__decay_depth': 0.1,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0,
        ...           'recharge_rate': 0.5,
        ...           'soil__initial_thickness': 2.0,
        ...           'hydraulic_conductivity': 0.1}

        Construct the model.

        >>> model = BasicSaVs(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """
        # Call ErosionModel's init
        super(BasicSaVs, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        # Get Parameters and convert units if necessary:
        self.m = self.params["m_sp"]
        self.n = self.params["n_sp"]
        self.K = self.get_parameter_from_exponent("water_erodability") * (
            self._length_factor ** (1. - (2. * self.m))
        )

        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        initial_soil_thickness = (self._length_factor) * self.params[
            "soil__initial_thickness"
        ]  # has units length

        soil_transport_decay_depth = (self._length_factor) * self.params[
            "soil_transport_decay_depth"
        ]  # has units length
        max_soil_production_rate = (self._length_factor) * self.params[
            "soil_production__maximum_rate"
        ]  # has units length per time
        soil_production_decay_depth = (self._length_factor) * self.params[
            "soil_production__decay_depth"
        ]  # has units length

        recharge_rate = (self._length_factor) * self.params[
            "recharge_rate"
        ]  # has units length per time
        K_hydraulic_conductivity = (self._length_factor) * self.params[
            "hydraulic_conductivity"
        ]  # has units length per time

        # Create soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.add_zeros("node", "soil__depth")

        # Create bedrock elevation field
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")

        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Add a field for effective drainage area
        self.eff_area = self.grid.add_zeros("node", "effective_drainage_area")

        # Get the effective-length parameter
        self.sat_len = (K_hydraulic_conductivity * self.grid.dx) / (recharge_rate)

        # Instantiate a FastscapeEroder component
        self.eroder = StreamPowerEroder(
            self.grid, use_Q=self.eff_area, K_sp=self.K, m_sp=self.m, n_sp=self.n
        )

        # Instantiate a DepthDependentDiffuser component
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

    def _calc_effective_drainage_area(self):
        """Calculate and store effective drainage area."""
        area = self.grid.at_node["drainage_area"]
        slope = self.grid.at_node["topographic__steepest_slope"]
        soil = self.grid.at_node["soil__depth"]
        cores = self.grid.core_nodes
        self.eff_area[cores] = area[cores] * (
            np.exp(-self.sat_len * soil[cores] * slope[cores] / area[cores])
        )

    def run_one_step(self, dt):
        """Advance model **BasicVs** for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow, accumulates drainage area, and calculates effective
           drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a **PrecipChanger** is an active BoundaryHandler and if
           so, uses it to modify the two erodability by water values.

        4. Calculates detachment-limited erosion by water.

        5. Calculates topographic change by linear diffusion.

        6. Finalizes the step using the **ErosionModel** base class function
           **finalize__run_one_step**. This function updates all BoundaryHandlers
           by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """
        # Direct and accumulate flow
        self.flow_accumulator.run_one_step()

        # Update effective runoff ratio
        self._calc_effective_drainage_area()

        # Get IDs of flooded nodes, if any
        if self.flow_accumulator.depression_finder is None:
            flooded = []
        else:
            flooded = np.where(
                self.flow_accumulator.depression_finder.flood_status == 3
            )[0]

        # Zero out effective area in flooded nodes
        self.eff_area[flooded] = 0.0

        # Do some erosion (but not on the flooded nodes)
        # (if we're varying K through time, update that first)
        if "PrecipChanger" in self.boundary_handler:
            self.eroder.K = (
                self.K_sp
                * self.boundary_handler[
                    "PrecipChanger"
                ].get_erodability_adjustment_factor()
            )
        self.eroder.run_one_step(dt)

        # We must also now erode the bedrock where relevant. If water erosion
        # into bedrock has occurred, the bedrock elevation will be higher than
        # the actual elevation, so we simply re-set bedrock elevation to the
        # lower of itself or the current elevation.
        b = self.grid.at_node["bedrock__elevation"]
        b[:] = np.minimum(b, self.grid.at_node["topographic__elevation"])

        # Calculate regolith-production rate
        self.weatherer.calc_soil_prod_rate()

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

    vssa = BasicVsSa(input_file=infile)
    vssa.run()


if __name__ == "__main__":
    main()
