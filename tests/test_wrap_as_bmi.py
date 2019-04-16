import sys
import warnings

import pytest

from terrainbento.derived_models import MODELS


@pytest.mark.parametrize("Model", MODELS)
def test_wrap_as_bmi(Model):
    # test something

    # verify  that all models BMI attributes are the correct values in the
    # docstring information.

    # verify that all units are either None or UDUNITS compatible.

    pass
