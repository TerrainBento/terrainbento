import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import BasicChRt


def test_diffusion_only():
    U = 0.0005
    m = 0.5
    n = 1.0
    step = 2
    D = 1.0
    S_c = 0.3
    dx = 10.0
    runtime = 30000

    file_name = os.path.join(_TEST_DATA_DIR, "example_contact_diffusion.asc")

    # Construct dictionary. Note that stream power is turned off
    params = {
        "model_grid": "RasterModelGrid",
        "clock": {"step": step, "output_interval": 2., "stop": 200.},
        "number_of_node_rows": 21,
        "number_of_node_columns": 3,
        "node_spacing": dx,
        "east_boundary_closed": True,
        "west_boundary_closed": True,
        "regolith_transport_parameter": D,
        "water_erodability_lower": 0,
        "water_erodability_upper": 0,
        "contact_zone__width": 1.0,
        "m_sp": m,
        "n_sp": n,
        "critical_slope": S_c,
        "lithology_contact_elevation__file_name": file_name,
        "depression_finder": "DepressionFinderAndRouter",
        "BoundaryHandlers": "NotCoreNodeBaselevelHandler",
        "NotCoreNodeBaselevelHandler": {"modify_core_nodes": True, "lowering_rate": -U},
    }

    # Construct and run model
    model = BasicChRt(params=params)
    for _ in range(runtime):
        model.run_one_step(step)

    # Construct actual and predicted slope at top edge of domain
    x = 8.5 * dx
    qs = U * x
    nterms = 11
    p = np.zeros(2 * nterms - 1)
    for k in range(1, nterms + 1):
        p[2 * k - 2] = D * (1 / (S_c ** (2 * (k - 1))))
    p = np.fliplr([p])[0]
    p = np.append(p, qs)
    p_roots = np.roots(p)
    predicted_slope = np.abs(np.real(p_roots[-1]))
    # print(predicted_slope)

    actual_slope = np.abs(model.grid.at_node["topographic__steepest_slope"][7])
    # print model.grid.at_node["topographic__steepest_slope"]
    assert_array_almost_equal(actual_slope, predicted_slope, decimal=3)
