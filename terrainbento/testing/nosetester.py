from __future__ import print_function

import os
import sys
from numpy.testing import Tester
import nose
from nose.plugins import doctests
from nose.plugins.base import Plugin

import terrainbento

def show_system_info():
    print('terrainbento version %s' % terrainbento.__version__)
    terrainbento_dir = os.path.dirname(terrainbento.__file__)
    print('terrainbento is installed in %s' % terrainbento_dir)

    print('Python version %s' % sys.version.replace('\n', ''))
    print('nose version %d.%d.%d' % nose.__versioninfo__)


class TerrainBentoDoctest(doctests.Doctest):
    name = 'terrainbentodoctest'
    score = 1000
    doctest_ignore = ('setup.py', )

    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
        self.doctest_tests = True
        self._doctest_result_var = None

    def configure(self, options, config):
        options.doctestOptions = ["+ELLIPSIS", "+NORMALIZE_WHITESPACE"]
        super(TerrainBentoDoctest, self).configure(options, config)


class TerrainBentoTester(Tester):
    excludes = ['examples']

    def __init__(self, package=None, raise_warnings='develop'):
        package_name = None
        if package is None:
            f = sys._getframe(1)
            package_path = f.f_locals.get('__file__', None)
            if package_path is None:
                raise AssertionError
            package_path = os.path.dirname(package_path)
            package_name = f.f_locals.get('__name__', None)
        elif isinstance(package, type(os)):
            package_path = os.path.dirname(package.__file__)
            package_name = getattr(package, '__name__', None)
        else:
            package_path = str(package)

        self.package_path = os.path.abspath(package_path)

        # Find the package name under test; this name is used to limit coverage
        # reporting (if enabled).
        if package_name is None:
            #package_name = get_package_name(package_path)
            package_name = 'terrainbento'
        self.package_name = package_name

        # Set to "release" in constructor in maintenance branches.
        self.raise_warnings = raise_warnings

    def _get_custom_doctester(self):
        return TerrainBentoDoctest()

    def test(self, **kwds):
        kwds.setdefault('label', 'fast')
        kwds.setdefault('verbose', 1)
        kwds.setdefault('doctests', True)
        kwds.setdefault('coverage', False)
        kwds.setdefault('extra_argv', [])
        kwds.setdefault('raise_warnings', 'release')
        show_system_info()
        return super(TerrainBentoTester, self).test(**kwds)
