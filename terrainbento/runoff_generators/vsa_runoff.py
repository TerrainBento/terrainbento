"""terrainbento **VariableSourceAreaRunoff**."""

import numpy as np


class VariableSourceAreaRunoff(object):
    r"""Generate variable source area runoff.

    **VariableSourceAreaRunoff** populates the field "water__unit_flux_in"
    with a value
    proportional to the "rainfall__flux".

    The "water__unit_flux_in" field is accumulated to create discharge.

    In this simple version of variable source area hydrology, discharge
    (:math:`Q`) is given as a function of precipitation :math:`P` and an
    effective area :math:`A_{eff}`.

    .. math::

        Q = P A_{eff}

    The effective area is defined as

    .. math::

        A_{eff} = A e^{\frac{-K_s H \Delta x S}{P A}}

    where :math:`K_s` is the hydraulic conductivity, :math:`H` is the soil
    thickness, :math:`\Delta x` is the grid cell width, :math:`S` is the local
    slope, and :math:`A` is the drainage area.

    This can be re-cast as

    .. math::

        Q = \int_0^A R_0 P dA = P A e^{\frac{-K_s H \Delta x S(A)}{P A}}

    where :math:`R_0` is a runoff fraction.

    Take that

    .. math::

        c = \frac{-K_s H \Delta x}{P} \;.

    Then

    .. math::

        \int_0^A R_0 dA = A e^{\frac{-c S(A)}{A}} \;.

    Solving for :math:`R_0` using the chain rule we get

    .. math::

        R_0(A) = \left ( \frac{c}{A} + 1 \right) e^{\frac{-c}{A}} \;.


    We must use the average value of :math:`R_0` over the entire cell as
    :math`R_0` varies non-linearly with A.

    If we ignore the fact that :math:`S` varies over space (and use only the
    local value), we get the runnoff from a cell. We do not get contributions
    from subsurface flow that consider that streams turn from losing streams to
    gaining streams when they flatten out and the subsurface flow capacity
    decreases.

    Examples
    --------
    Start by importing necessary components and functions.

    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowAccumulator
    >>> from landlab.values import plane
    >>> from terrainbento import UniformPrecipitator, VariableSourceAreaRunoff

    Create a grid with elevation and soil depth.

    >>> grid = RasterModelGrid((5,5))
    >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, False)
    >>> H = grid.add_ones("node", "soil__depth")
    >>> z = grid.add_field(
    ...     'topographic__elevation',
    ...     grid.node_x + grid.node_y,
    ...     at = 'node'
    ...     )
    >>> z.reshape(grid.shape)
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 1.,  2.,  3.,  4.,  5.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 3.,  4.,  5.,  6.,  7.],
           [ 4.,  5.,  6.,  7.,  8.]])

    In terrainbento models the **VariableSourceAreaRunoff** will be run as part
    of a model that does flow accumulation. Here, for simplicity we will run it
    on its own. However, the **VariableSourceAreaRunoff** requires the drainage
    area and slope to calculate runoff. Thus we will have to do flow
    accumulation explicitly.

    >>> flow_accumulator = FlowAccumulator(grid)
    >>> flow_accumulator.run_one_step()
    >>> grid.at_node["drainage_area"].reshape(grid.shape)
    array([[ 0.,  3.,  3.,  3.,  0.],
           [ 0.,  3.,  3.,  3.,  0.],
           [ 0.,  2.,  2.,  2.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    >>> grid.at_node["topographic__steepest_slope"].reshape(grid.shape)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    Create a precipitator with uniform rainfall flux of 1.0.

    >>> precipitator = UniformPrecipitator(grid)
    >>> precipitator.run_one_step(10)
    >>> grid.at_node["rainfall__flux"].reshape(grid.shape)
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])

    Now the grid has all the fields the **VariableSourceAreaRunoff**
    runoff_generator expects. Create an instance with a hydraulic conductivity,
    :math:K_s, of 0.2.

    >>> Ks = 0.2
    >>> runoff_generator = VariableSourceAreaRunoff(
    ...     grid,
    ...     hydraulic_conductivity = Ks)

    Run the runoff generator in order to calculate the value of runoff (which
    is stored in the field "water__unit_flux_in").

    >>> runoff_generator.run_one_step(10)
    >>> grid.at_node["water__unit_flux_in"].reshape(grid.shape)
    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.99684612,  0.99684612,  0.99684612,  0.        ],
           [ 0.        ,  0.99094408,  0.99094408,  0.99094408,  0.        ],
           [ 0.        ,  0.81873075,  0.81873075,  0.81873075,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    Next this runoff is accumulated.

    >>> flow_accumulator = FlowAccumulator(grid)
    >>> flow_accumulator.run_one_step()
    >>> grid.at_node["surface_water__discharge"].reshape(grid.shape)
    array([[ 0.        ,  2.80652094,  2.80652094,  2.80652094,  0.        ],
           [ 0.        ,  2.80652094,  2.80652094,  2.80652094,  0.        ],
           [ 0.        ,  1.80967486,  1.80967486,  1.80967486,  0.        ],
           [ 0.        ,  0.81873075,  0.81873075,  0.81873075,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    Finally, we compare the results of the **VariableSourceAreaRunoff** with
    the theoretical values for discharge (:math:`Q`).

    >>> import numpy as np
    >>> P = grid.at_node["rainfall__flux"]
    >>> A = grid.at_node["drainage_area"]
    >>> S = grid.at_node["topographic__steepest_slope"]

    >>> Q = P * A * np.exp( -(Ks * H * grid.dx * S) / (P * A) )
    >>> Q
    array([        nan,  3.        ,  3.        ,  3.        ,         nan,
                   nan,  2.80652096,  2.80652096,  2.80652096,         nan,
                   nan,  1.80967484,  1.80967484,  1.80967484,         nan,
                   nan,  0.81873075,  0.81873075,  0.81873075,         nan,
                   nan,         nan,         nan,         nan,         nan])

    >>> from numpy.testing import assert_array_almost_equal
    >>> assert_array_almost_equal(
    ...     grid.at_node["surface_water__discharge"][grid.core_nodes],
    ...     Q[grid.core_nodes])

    This example works because the slope is constant.

    """

    def __init__(self, grid, hydraulic_conductivity=0.2):
        """
        Parameters
        ----------
        grid : model grid
        hydraulic_conductivity : float, optional.
            Hydraulic conductivity. Default is 0.2.
        """
        self._grid = grid

        if "water__unit_flux_in" not in grid.at_node:
            grid.add_ones("node", "water__unit_flux_in")  # line not yet tested

        self._hydraulic_conductivity = hydraulic_conductivity

    def run_one_step(self, step):
        """Run **VariableSourceAreaRunoff** forward by duration ``step``"""
        self._p = self._grid.at_node["rainfall__flux"]
        self._area = self._grid.at_node["drainage_area"]
        self._slope = self._grid.at_node["topographic__steepest_slope"]
        self._H = self._grid.at_node["soil__depth"]

        # Get the transmissivity parameter
        # transmissivity is hydraulic condiuctivity times soil thickness
        self._transmissivity = self._hydraulic_conductivity * self._H
        if np.any(self._transmissivity) <= 0.0:
            raise ValueError(
                "VariableSourceAreaRunoff: Transmissivity must be > 0"
            )  # line not yet tested

        a = self._transmissivity * self._grid.dx * self._slope / self._p

        runoff_coefficient = (
            _definite_integral(self._grid.at_node["drainage_area"], a)
            - _definite_integral(
                (
                    self._grid.at_node["drainage_area"]
                    - self._grid.cell_area_at_node
                ),
                a,
            )
        ) / self._grid.cell_area_at_node

        runoff_coefficient[np.isnan(runoff_coefficient)] = 0.0

        self._grid.at_node["water__unit_flux_in"][:] = 0.0
        self._grid.at_node["water__unit_flux_in"][self._grid.core_nodes] = (
            runoff_coefficient[self._grid.core_nodes]
            * self._p[self._grid.core_nodes]
        )


def _definite_integral(x, c):
    return x * np.exp(-c / x)
