#! /usr/bin/env python
from __future__ import print_function

import sys
import os
#import coverage


sys.path.pop(0)

#cov = coverage.Coverage()
#cov.start()

from optparse import OptionParser

parser = OptionParser('usage: %prog [options] -- [nosetests options]')
parser.add_option('-v', '--verbose', action='count', dest='verbose',
                  default=1, help='increase verbosity [%default]')
parser.add_option('--no-doctests', action='store_false', dest='doctests',
                  default=True,
                  help='Do not run doctests in module [%default]')
parser.add_option('--coverage', action='store_true', dest='coverage',
                   default=False, help='report coverage of terrainbento [%default]')
parser.add_option('-m', '--mode', action='store', dest='mode', default='fast',
                  help='"fast", "full", or something that can be passed to '
                  'nosetests -A [%default]')

(options, args) = parser.parse_args()
try:
    import terrainbento
except ImportError:
    print('Unable to import terrainbento. You may not have terrainbento installed.')
    print('Here is your sys.path')
    print(os.linesep.join(sys.path))
    raise


result = terrainbento.test(label=options.mode, verbose=options.verbose,
                           doctests=options.doctests, coverage=options.coverage,
                           extra_argv=args, raise_warnings='release')

#cov.stop()
#cov.save()

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
