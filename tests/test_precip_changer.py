import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import NotCoreNodeBaselevelHandler, PrecipChanger, Basic, BasicCh, BasicChSa, BasicSa


@pytest.mark.parametrize("Model", [Basic, BasicCh, BasicChSa, BasicSa])
def test_with_precip_changer(
    clock_simple, grid_1, precip_defaults, precip_testing_factor, K, Model
):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.,
        "water_erodability": K,
        "m_sp": 0.5,
        "n_sp": 1.0,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }
    model = Model(**params)
    assert model.eroder.K == K
    assert "PrecipChanger" in model.boundary_handlers
    model.run_one_step(1.0)
    model.run_one_step(1.0)
    assert round(model.eroder.K, 5) == round(K * precip_testing_factor, 5)
