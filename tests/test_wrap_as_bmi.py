import sys
import warnings

import pytest

from terrainbento.derived_models import MODELS

from .bmi_bridge import wrap_as_bmi


@pytest.mark.parameterize(Model, Models)
def test_wrap_as_bmi():
    wrap_as_bmi(Model)
