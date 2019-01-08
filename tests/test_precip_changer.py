import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    NotCoreNodeBaselevelHandler,
    PrecipChanger,
    Basic,
    BasicCh,
    BasicChSa,
    BasicSa,
    BasicRt,
    BasicChRt,
    BasicChRtTh,
    BasicHyRt,
    BasicRtSa,
    BasicRtTh,
    BasicRtVs,
    BasicDdRt,
)


@pytest.mark.parametrize("Model", [Basic, BasicCh, BasicChSa, BasicSa])
def test_simple_precip_changer(
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


@pytest.mark.parametrize(
    "Model",
    [
        BasicRt,
        BasicChRt,
        BasicChRtTh,
        BasicHyRt,
        BasicRtSa,
        BasicRtTh,
        BasicRtVs,
        BasicDdRt,
    ],
)
def test_rock_till_precip_changer(
    clock_simple, grid_3, precip_defaults, precip_testing_factor, Kt, Kr, Model
):
    precip_changer = PrecipChanger(grid_3, **precip_defaults)
    params = {
        "grid": grid_3,
        "clock": clock_simple,
        "water_erodability_lower": Kr,
        "water_erodability_upper": Kt,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }

    model = Model(**params)
    model._update_erodability_field()
    try:
        assert np.array_equiv(model.eroder.K[model.grid.core_nodes[:8]], Kt) is True
        assert np.array_equiv(model.eroder.K[model.grid.core_nodes[10:]], Kr) is True
    except AttributeError:
        assert (
            np.array_equiv(model.eroder._K_unit_time[model.grid.core_nodes[:8]], Kt)
            is True
        )
        assert (
            np.array_equiv(model.eroder._K_unit_time[model.grid.core_nodes[10:]], Kr)
            is True
        )

    assert "PrecipChanger" in model.boundary_handlers
    model.run_one_step(1.0)
    model.run_one_step(1.0)

    try:
        assert_array_almost_equal(
            model.eroder.K[model.grid.core_nodes[:8]],
            Kt * precip_testing_factor * np.ones((8)),
        )
        assert_array_almost_equal(
            model.eroder.K[model.grid.core_nodes[10:]],
            Kr * precip_testing_factor * np.ones((9)),
        )
    except AttributeError:
        assert_array_almost_equal(
            model.eroder._K_unit_time[model.grid.core_nodes[:8]],
            Kt * precip_testing_factor * np.ones((8)),
        )
        assert_array_almost_equal(
            model.eroder._K_unit_time[model.grid.core_nodes[10:]],
            Kr * precip_testing_factor * np.ones((9)),
        )
