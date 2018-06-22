#! /usr/bin/env python
from __future__ import print_function

import sys
import os

import pytest

sys.path.pop(0)

try:
    import terrainbento
except ImportError:
    print('Unable to import terrainbento. You may not have terrainbento installed.')
    print('Here is your sys.path')
    print(os.linesep.join(sys.path))
    raise


result = pytest.main()

if result == 0:
    sys.exit(0)
else:
    sys.exit(1)
