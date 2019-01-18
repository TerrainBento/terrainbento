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

        Q = \int_0^A R_0 P dA = P A e^{\frac{-K_s H \Delta x S}{P A}}

    where :math:`R_0` is a runoff fraction.

    Take that

    .. math::

        c = \frac{-K_s H \Delta x S}{P} \;.

    Then

    .. math::

        \int_0^A R_0 dA = A e^{\frac{-c}{A}} \;.

    Solving for :math:`R_0` using the chain rule we get

    .. math::

        R_0(A) = \left ( \frac{c}{A} + 1 \right) e^{\frac{-c}{A}} \;.


    This doesn't end up working out. Some ideas as to why:
        1) Does the lower limit of the integral need to coorespond to the
        minimum area/slope combination at which discharge is produced?

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
    ...     grid.node_x**2 + grid.node_y,
    ...     at = 'node'
    ...     )
    >>> z.reshape(grid.shape)
    array([[  0.,   1.,   4.,   9.,  16.],
           [  1.,   2.,   5.,  10.,  17.],
           [  2.,   3.,   6.,  11.,  18.],
           [  3.,   4.,   7.,  12.,  19.],
           [  4.,   5.,   8.,  13.,  20.]])

    In terrainbento models the **VariableSourceAreaRunoff** will be run as part
    of a model that does flow accumulation. Here, for simplicity we will run it
    on its own. However, the **VariableSourceAreaRunoff** requires the drainage
    area and slope to calculate runoff. Thus we will have to do flow
    accumulation explicitly.

    >>> flow_accumulator = FlowAccumulator(grid)
    >>> flow_accumulator.run_one_step()
    >>> grid.at_node["drainage_area"].reshape(grid.shape)
    array([[ 0.,  9.,  0.,  0.,  0.],
           [ 0.,  9.,  2.,  1.,  0.],
           [ 0.,  6.,  2.,  1.,  0.],
           [ 0.,  3.,  2.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    >>> grid.at_node["topographic__steepest_slope"].reshape(grid.shape)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  3.,  5.,  0.],
           [ 0.,  1.,  3.,  5.,  0.],
           [ 0.,  1.,  3.,  5.,  0.],
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
           [ 0.        ,  0.99975671,  0.96306369,  0.73575888,  0.        ],
           [ 0.        ,  0.99945664,  0.96306369,  0.73575888,  0.        ],
           [ 0.        ,  0.99787412,  0.96306369,  0.73575888,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    Next this runoff is accumulated.

    >>> flow_accumulator = FlowAccumulator(grid)
    >>> flow_accumulator.run_one_step()
    >>> grid.at_node["surface_water__discharge"].reshape(grid.shape)
    array([[ 0.        ,  8.09355545,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  8.09355545,  1.69882262,  0.73575888,  0.        ],
           [ 0.        ,  5.39497614,  1.69882262,  0.73575888,  0.        ],
           [ 0.        ,  2.69669676,  1.69882262,  0.73575888,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    Finally, we compare the results of the **VariableSourceAreaRunoff** with
    the theoretical values for discharge (:math:`Q`).

    >>> import numpy as np
    >>> P = grid.at_node["rainfall__flux"]
    >>> A = grid.at_node["drainage_area"]
    >>> S = grid.at_node["topographic__steepest_slope"]

    >>> Q = P * A * np.exp( -(Ks * H * grid.dx * S) / (P * A) )
    >>> Q
    array([        nan,  9.        ,         nan,         nan,         nan,
                   nan,  8.80220585,  1.48163644,  0.36787944,         nan,
                   nan,  5.8032966 ,  1.48163644,  0.36787944,         nan,
                   nan,  2.80652096,  1.48163644,  0.36787944,         nan,
                   nan,         nan,         nan,         nan,         nan])
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

        runoff_coefficient = ((a / self._area) + 1.0) * np.exp(
            -(a / self._area)
        )

        runoff_coefficient[np.isnan(runoff_coefficient)] = 0.0

        self._grid.at_node["water__unit_flux_in"][:] = 0.0
        self._grid.at_node["water__unit_flux_in"][self._grid.core_nodes] = (
            runoff_coefficient[self._grid.core_nodes]
            * self._p[self._grid.core_nodes]
        )
