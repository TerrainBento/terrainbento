import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChRt, BasicChRtTh, NotCoreNodeBaselevelHandler


@pytest.mark.parametrize("Model", [BasicChRt, BasicChRtTh])
def test_diffusion_only(clock_09, grid_4, Model):
    U = 0.0005
    D = 1.0
    S_c = 0.3

    ncnblh = NotCoreNodeBaselevelHandler(
        grid_4, modify_core_nodes=True, lowering_rate=-U
    )

    params = {
        "grid": grid_4,
        "clock": clock_09,
        "regolith_transport_parameter": D,
        "water_erodibility_lower": 0,
        "water_erodibility_upper": 0,
        "critical_slope": S_c,
        "boundary_handlers": {"NotCoreNodeBaselevelHandler": ncnblh},
    }

    # Construct and run model
    model = Model(**params)
    for _ in range(20000):
        model.run_one_step(clock_09.step)

    # Construct actual and predicted slope at right edge of domain
    x = 8.5 * grid_4.dx

    qs = U * x
    nterms = 7
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = D * (1 / (S_c ** (2 * (k - 1))))
    p = np.fliplr([p])[0]
    p = np.append(p, qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    # print(predicted_slope)

    actual_slope = np.abs(
        model.grid.at_node["topographic__steepest_slope"][39]
    )
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=2)
