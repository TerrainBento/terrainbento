import glob
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_equal

from landlab import RasterModelGrid
from terrainbento import BasicSt, BasicStTh, Clock, NotCoreNodeBaselevelHandler

_th_params = {
    "water_erosion_rule__threshold": 1e-9,
    "infiltration_capacity": 1.0,
}
_empty_params = {"infiltration_capacity": 1.0}


@pytest.mark.parametrize(
    "Model,extra_params", [(BasicStTh, _th_params), (BasicSt, _empty_params)]
)
@pytest.mark.parametrize(
    "depression_finder", [None, "DepressionFinderAndRouter"]
)
def test_steady_without_stochastic_duration(
    clock_simple, Model, extra_params, depression_finder
):
    r"""Test steady profile solution with fixed duration.

    Notes
    -----
    We use m=1 because the integral that averages over storm events has an
    analytical solution; it evaluates to 1/2 when mean rain intensity and
    infiltration capacity are both equal to unity. This is where the factor of
    2 in the predicted-slope calculation below comes from.

    The derivation is as follows.

    Instantaneous erosion rate, :math:E_i:

    ..math::

        E_i = K Q^m S^n

    Instantaneous water discharge depends on drainage area, :math:A, rain
    intensity, :math:P, and infiltration capacity, :math:I_m:

    ..math::

        Q = R A
        R = P - I_m (1 - e^{-P/I_m})

    Average erosion rate, :math:E, is the integral of instantaneous erosion
    rate over all possible rain rates times the PDF of rain rate, :math:f(P):

    ..math::

        E = \int_0^\infty f(P) K A^m S^n [P-I_m(1-e^{-P/I_m})]^m dP
          = K A^m S^n \int_0^\infty f(P) [P-I_m(1-e^{-P/I_m})]^m dP
          = K A^m S^n \Phi

    where :math:\Phi represents the integral. For testing purposes, we seek an
    analytical solution to the integral. Take :math:`m=n=1` and :math:`P=I_m=1`. Also
    assume that the distribution shape factor is 1, so that
    :math:`f(P) = (1/Pbar) e^{-P/Pbar}`.

    According to the online integrator, the indefinite integral solution under
    these assumptions is

    ..math::

        \Phi = e^{-P} (-\frac{1}{2} e^{-P} - P)

    The definite integral should therefore be 1/2.

    The slope-area relation is therefore

    ..math::

        S = \frac{2U}{K A}
    """
    U = 0.0001
    K = 0.001
    m = 1.0
    n = 1.0

    grid = RasterModelGrid((3, 6), xy_spacing=100.0)
    grid.set_closed_boundaries_at_grid_edges(False, True, False, True)
    grid.add_zeros("node", "topographic__elevation")
    s = grid.add_zeros("node", "soil__depth")
    s[:] = 1e-9

    ncnblh = NotCoreNodeBaselevelHandler(
        grid, modify_core_nodes=True, lowering_rate=-U
    )

    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "m_sp": m,
        "n_sp": n,
        "number_of_sub_time_steps": 100,
        "rainfall_intermittency_factor": 1.0,
        "rainfall__mean_rate": 1.0,
        "rainfall__shape_factor": 1.0,
        "random_seed": 3141,
        "depression_finder": depression_finder,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    for p in extra_params:
        params[p] = extra_params[p]

    # construct and run model
    model = Model(**params)
    for _ in range(100):
        model.run_one_step(1.0)

    # construct actual and predicted slopes
    ic = model.grid.core_nodes[1:-1]  # "inner" core nodes
    actual_slopes = model.grid.at_node["topographic__steepest_slope"][ic]
    actual_areas = model.grid.at_node["drainage_area"][ic]
    predicted_slopes = 2 * U / (K * (actual_areas))

    # assert actual and predicted slopes are the same.
    assert_array_almost_equal(actual_slopes, predicted_slopes)


def test_stochastic_duration_rainfall_means():
    """Test option with stochastic duration.

    Test is simply to get the correct total cumulative rain depth.
    """
    U = 0.0001
    K = 0.0001
    m = 1.0
    n = 1.0

    grid = RasterModelGrid((3, 6), xy_spacing=100.0)
    grid.set_closed_boundaries_at_grid_edges(True, False, True, False)
    grid.add_zeros("node", "topographic__elevation")

    ncnblh = NotCoreNodeBaselevelHandler(
        grid, modify_core_nodes=True, lowering_rate=-U
    )

    clock = Clock(step=200, stop=400)
    # construct dictionary. note that D is turned off here
    params = {
        "grid": grid,
        "clock": clock,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "m_sp": m,
        "n_sp": n,
        "opt_stochastic_duration": True,
        "record_rain": True,
        "mean_storm_duration": 1.0,
        "mean_interstorm_duration": 1.0,
        "infiltration_capacity": 1.0,
        "random_seed": 3141,
        "mean_storm_depth": 1.0,
        "output_interval": 401,
        "depression_finder": "DepressionFinderAndRouter",
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # construct and run model
    model = BasicSt(**params)
    model.run()

    cum_rain_depth = np.sum(
        np.array(model.rain_record["event_duration"])
        * np.array(model.rain_record["rainfall_rate"])
    )
    assert_equal(np.round(cum_rain_depth), 200.0)

    os.remove("storm_sequence.txt")
    fs = glob.glob(model._out_file_name + "*.nc")
    for f in fs:
        os.remove(f)


#
