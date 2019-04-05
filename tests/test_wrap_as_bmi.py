import sys
import warnings

import pytest

from terrainbento.derived_models import MODELS


@pytest.mark.parametrize("Model", MODELS)
def test_wrap_as_bmi(Model):
    # test something
    pass
