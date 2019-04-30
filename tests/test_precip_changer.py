import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from terrainbento import (
    Basic,
    BasicCh,
    BasicChRt,
    BasicChRtTh,
    BasicChSa,
    BasicDd,
    BasicDdHy,
    BasicDdRt,
    BasicDdVs,
    BasicHy,
    BasicHyRt,
    BasicHyVs,
    BasicRt,
    BasicRtSa,
    BasicRtTh,
    BasicRtVs,
    BasicSa,
    BasicSaVs,
    BasicTh,
    BasicThVs,
    BasicVs,
    PrecipChanger,
)


@pytest.mark.parametrize(
    "Model",
    [
        Basic,
        BasicCh,
        BasicChSa,
        BasicSa,
        BasicHy,
        BasicDdHy,
        BasicDd,
        BasicDdVs,
        BasicTh,
        BasicThVs,
        BasicHyVs,
        BasicVs,
        BasicSaVs,
    ],
)
def test_simple_precip_changer(
    clock_simple, grid_1, precip_defaults, precip_testing_factor, K, Model
):
    precip_changer = PrecipChanger(grid_1, **precip_defaults)
    params = {
        "grid": grid_1,
        "clock": clock_simple,
        "regolith_transport_parameter": 0.0,
        "water_erodibility": K,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }
    model = Model(**params)
    try:
        assert model.eroder.K == K
    except ValueError:
        assert model.eroder.K[0] == K

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
        "water_erodibility_lower": Kr,
        "water_erodibility_upper": Kt,
        "boundary_handlers": {"PrecipChanger": precip_changer},
    }

    model = Model(**params)
    model._update_erodibility_field()

    assert (
        np.array_equiv(model.eroder.K[model.grid.core_nodes[:8]], Kt) is True
    )
    assert (
        np.array_equiv(model.eroder.K[model.grid.core_nodes[10:]], Kr) is True
    )

    assert "PrecipChanger" in model.boundary_handlers
    model.run_one_step(1.0)
    model.run_one_step(1.0)

    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[:8]],
        Kt * precip_testing_factor * np.ones((8)),
    )
    assert_array_almost_equal(
        model.eroder.K[model.grid.core_nodes[10:]],
        Kr * precip_testing_factor * np.ones((9)),
    )
