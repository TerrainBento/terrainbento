"""
Wrap terrainbento model with a bmi
----------------------------------

The `wrap_as_bmi` function wraps a terrainbento model class so that it
exposes a Basic Modelling Interface.

"""
import os

import numpy as np
import yaml

from terrainbento.base_class.model import Model


def wrap_as_bmi(cls):
    """Wrap a landlab class so it exposes a BMI.

    Parameters
    ----------
    cls : class
        A terrainbento class that inherits from `Model`. TODO: link.

    Returns
    -------
    class
        A wrapped class that exposes a BMI.

    Examples
    --------
    >>> from terrainbento.bmi import wrap_as_bmi
    >>> from terrainbento import Basic

    >>> BmiBasic = wrap_as_bmi(Basic)
    >>> basic = BmiBasic()

    >>> config = \"\"\"
    ... clock:
    ...     start: 0.
    ...     stop: 10.
    ...     step: 2.
    ... grid:
    ...     type: raster
    ...     shape: [20, 40]
    ...     spacing: [1000., 2000.]
    ... \"\"\"
    >>> basic.initialize(config)
    >>> basic.get_output_var_names()
    ('topographic__elevation',)
    >>> basic.get_var_grid('topographic__elevation')
    0
    >>> basic.get_grid_shape(0)
    (20, 40)
    >>> dz = basic.get_value('topographic__elevation')
    >>> dz.shape == (800, )
    True

    >>> np.all(dz == 0.)
    True
    >>> basic.get_current_time()
    0.0

    >>> basic.get_input_var_names()
    ('topographic__elevation',)
    >>> z = np.zeros((20, 40), dtype=float)
    >>> z[0, 0] = 1.
    >>> basic.set_value('topographic__elevation', z)
    >>> basic.update()
    >>> basic.get_current_time()
    2.0
    >>> dz = basic.get_value('topographic__elevation')
    >>> np.all(dz == 0.)
    False

    >>> some examples with parameter values.
    """
    if not issubclass(cls, Model):
        raise TypeError("class must inherit from Model")

    class BmiWrapper(object):
        __doc__ = """
        Basic Modeling Interface for the {name} model.
        """.format(
            name=cls.__name__
        ).strip()

        _cls = cls

        def __init__(self):
            self._base = None
            super(BmiWrapper, self).__init__()

    BmiWrapper.__name__ = cls.__name__
    return BmiWrapper
