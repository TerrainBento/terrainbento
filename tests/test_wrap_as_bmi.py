import sys
import warnings

import cfunits
import pytest

from terrainbento.derived_models import MODELS


@pytest.mark.parametrize("Model", MODELS)
def test_wrap_as_bmi(Model):

    # verify  that all models BMI attributes are the correct values in the
    # docstring information.

    # verify that all the grid information is correct.

    # verify that the model can run as expected.

    # verify that all units are either None or UDUNITS compatible.

    pass
