import os
import subprocess
import numpy as np

from numpy.testing import assert_array_almost_equal # assert_array_equal,
import pytest

from landlab import HexModelGrid
from terrainbento import BasicCh

def test_no_slope_crit():
	params = {'model_grid': 'RasterModelGrid',
        ...           'dt': 1,
        ...           'output_interval': 2.,
        ...           'run_duration': 200.,
        ...           'number_of_node_rows' : 6,
        ...           'number_of_node_columns' : 9,
        ...           'node_spacing' : 10.0,
        ...           'regolith_transport_parameter': 0.001,
        ...           'water_erodability': 0.001,
        ...           'm_sp': 0.5,
        ...           'n_sp': 1.0}

        pytest.raises(ValueError,BasicCh,params=params)

