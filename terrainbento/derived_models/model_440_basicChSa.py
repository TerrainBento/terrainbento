# coding: utf8
#! /usr/env/python
"""``terrainbento`` Model ``BasicChSa`` program.

Erosion model program using depth-dependent cubic diffusion
with a soil layer, basic stream power, and discharge proportional to drainage
area.

Landlab components used:
    1. `FlowAccumulator <http://landlab.readthedocs.io/en/release/landlab.components.flow_accum.html>`_
    2. `DepressionFinderAndRouter <http://landlab.readthedocs.io/en/release/landlab.components.flow_routing.html#module-landlab.components.flow_routing.lake_mapper>`_ (optional)
    3. `FastscapeEroder <http://landlab.readthedocs.io/en/release/landlab.components.stream_power.html>`_
    4. `ExponentialWeatherer <http://landlab.readthedocs.io/en/release/_modules/landlab/components/weathering/exponential_weathering.html#ExponentialWeatherer>`_
    5. `DepthDependentTaylorDiffuser <http://landlab.readthedocs.io/en/release/_modules/landlab/components/depth_dependent_taylor_soil_creep/hillslope_depth_dependent_taylor_flux.html#DepthDependentTaylorDiffuser>`_

"""

import sys
import numpy as np

from landlab.components import (
    FastscapeEroder,
    DepthDependentTaylorDiffuser,
    ExponentialWeatherer,
)
from terrainbento.base_class import ErosionModel


class BasicChSa(ErosionModel):
    """
    Model **BasicSa** explicitly resolves a soil layer. This soil layer is
    produced by weathering that decays exponentially with soil thickness and
    hillslope transport is soil-depth dependent. Given a spatially varying soil
    thickness :math:`H` and a spatially varying bedrock elevation :math:`\eta_b`,
    model **BasicSa** evolves a topographic surface described by :math:`\eta`
    with the following governing equations:

    .. math::

        \eta = \eta_b + H

        \\frac{\partial H}{\partial t} = P_0 \exp (-H/H_s) - \delta (H) K A^{1/2} S -\\nabla q_h

        \\frac{\partial \eta_b}{\partial t} = -P_0 \exp (-H/H_s) - (1 - \delta (H) ) K A^{1/2} S

        q_h = -DS \left[ 1 + \left( \\frac{S}{S_c} \\right)^2 +  \left( \\frac{S}{S_c} \\right)^4 + ... \left( \\frac{S}{S_c} \\right)^{2(N-1)} \\right]

    where :math:`A` is the local drainage area, :math:`S` is the local slope,
    :math:`K` is the erodability by water, :math:`D` is the regolith transport
    parameter, :math:`H_s` is the sediment production decay depth, :math:`H_s`
    is the sediment production decay depth, :math:`P_0` is the maximum sediment
    production rate, and :math:`H_0` is the sediment transport decay depth.
    :math:`S_c` is the critical slope parameter and :math:`N` is the number of
    terms in the Taylor Series expansion. Presently :math:`N` is set at 11 and
    is not a user defined parameter.

    The function :math:`\delta (H)` is used to indicate that water erosion will
    act on soil where it exists, and on the underlying lithology where soil is
    absent. To achieve this, :math:`\delta (H)` is defined to equal 1 when
    :math:`H > 0` (meaning soil is present), and 0 if :math:`H = 0` (meaning the
    underlying parent material is exposed).

    Refer to the terrainbento manuscript Table XX (URL here) for parameter
    symbols, names, and dimensions.

    Model ``BasicChSa`` inherits from the ``terrainbento`` ``ErosionModel`` base
    class.

    +------------------+-----------------------------------+-----------------+
    | Parameter Symbol | Input File Parameter Name         | Value           |
    +==================+===================================+=================+
    |:math:`m`         | ``m_sp``                          | 0.5             |
    +------------------+-----------------------------------+-----------------+
    |:math:`n`         | ``n_sp``                          | 1               |
    +------------------+-----------------------------------+-----------------+
    |:math:`K`         | ``water_erodability``             | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`D`         | ``regolith_transport_parameter``  | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`H_{init}`  | ``soil__initial_thickness``       | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`P_{0}`     | ``soil_production__maximum_rate`` | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`H_{s}`     | ``soil_production__decay_depth``  | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`H_{0}`     | ``soil_transport__decay_depth``   | user specified  |
    +------------------+-----------------------------------+-----------------+
    |:math:`S_c`       | ``critical_slope``                | user specified  |
    +------------------+-----------------------------------+-----------------+


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
        BasicChSa : model object

        Examples
        --------
        This is a minimal example to demonstrate how to construct an instance
        of model ``BasicChSa``. Note that a YAML input file can be used instead of
        a parameter dictionary. For more detailed examples, including steady-
        state test examples, see the ``terrainbento`` tutorials.

        To begin, import the model class.

        >>> from terrainbento import BasicChSa

        Set up a parameters variable.

        >>> params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'initial_soil_thickness': 0.0,
        ...           'soil_transport_decay_depth': 0.2,
        ...           'soil_production__maximum_rate': 0.001,
        ...           'soil_production__decay_depth': 0.1,
        ...           'critical_slope': 0.2,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        Construct the model.

        >>> model = BasicChSa(params=params)

        Running the model with ``model.run()`` would create output, so here we
        will just run it one step.

        >>> model.run_one_step(1.)
        >>> model.model_time
        1.0

        """

        # Call ErosionModel's init
        super(BasicChSa, self).__init__(
            input_file=input_file,
            params=params,
            BoundaryHandlers=BoundaryHandlers,
            OutputWriters=OutputWriters,
        )

        self.K_sp = self.get_parameter_from_exponent("water_erodability")
        regolith_transport_parameter = (
            self._length_factor ** 2.
        ) * self.get_parameter_from_exponent(
            "regolith_transport_parameter"
        )  # has units length^2/time
        try:
            initial_soil_thickness = (self._length_factor) * self.params[
                "soil__initial_thickness"
            ]  # has units length
        except KeyError:
            initial_soil_thickness = 1.0  # default value
        soil_transport_decay_depth = (self._length_factor) * self.params[
            "soil_transport_decay_depth"
        ]  # has units length
        max_soil_production_rate = (self._length_factor) * self.params[
            "soil_production__maximum_rate"
        ]  # has units length per time
        soil_production_decay_depth = (self._length_factor) * self.params[
            "soil_production__decay_depth"
        ]  # has units length

        # Create soil thickness (a.k.a. depth) field
        soil_thickness = self.grid.add_zeros("node", "soil__depth")

        # Create bedrock elevation field
        bedrock_elev = self.grid.add_zeros("node", "bedrock__elevation")

        soil_thickness[:] = initial_soil_thickness
        bedrock_elev[:] = self.z - initial_soil_thickness

        # Instantiate a FastscapeEroder component
        self.eroder = FastscapeEroder(
            self.grid,
            K_sp=self.K_sp,
            m_sp=self.params["m_sp"],
            n_sp=self.params["n_sp"],
        )

        # Instantiate a weathering component
        self.weatherer = ExponentialWeatherer(
            self.grid,
            soil_production__maximum_rate=max_soil_production_rate,
            soil_production__decay_depth=soil_production_decay_depth,
        )

        # Instantiate a soil-transport component
        self.diffuser = DepthDependentTaylorDiffuser(
            self.grid,
            linear_diffusivity=regolith_transport_parameter,
            slope_crit=self.params["critical_slope"],
            soil_transport_decay_depth=soil_transport_decay_depth,
            nterms=11,
        )

    def run_one_step(self, dt):
        """Advance model ``BasicChSa`` for one time-step of duration dt.

        The **run_one_step** method does the following:

        1. Directs flow and accumulates drainage area.

        2. Assesses the location, if any, of flooded nodes where erosion should
           not occur.

        3. Assesses if a ``PrecipChanger`` is an active BoundaryHandler and if
           so, uses it to modify the erodability by water.

        4. Calculates detachment-limited erosion by water.

        5. Produces soil and calculates soil depth with exponential weathering.

        6. Calculates topographic change by depth-dependent nonlinear diffusion.

        7. Finalizes the step using the ``ErosionModel`` base class function
           **finalize__run_one_step**. This function updates all BoundaryHandlers
           by ``dt`` and increments model time by ``dt``.

        Parameters
        ----------
        dt : float
            Increment of time for which the model is run.
        """

        # Direct and accumulate flow
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
                ].get_erodability_adjustment_factor()
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

        # Do some soil creep
        self.diffuser.run_one_step(
            dt, dynamic_dt=True, if_unstable="raise", courant_factor=0.1
        )

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

    cdsp = BasicChSa(input_file=infile)
    cdsp.run()


if __name__ == "__main__":
    main()
