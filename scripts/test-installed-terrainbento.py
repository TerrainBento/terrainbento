#! /usr/bin/env python
from __future__ import print_function

import sys
import os
import coverage


sys.path.pop(0)

cov = coverage.Coverage()
cov.start()

try:
    import terrainbento
except ImportError:
    print('Unable to import terrainbento. You may not have terrainbento installed.')
    print('Here is your sys.path')
    print(os.linesep.join(sys.path))
    raise


result = terrainbento.test()

cov.stop()
cov.save()

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
