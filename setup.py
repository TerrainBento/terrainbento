
from setuptools import setup
import versioneer

import os

setup(name='terrainbento',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='TerrainBento suite of landscape evolution models',
      url='https://github.com/TerrainBento/terrainbento/',
      author='The TerrainBento Team',
      author_email='barnhark@colorado.edu',
      license='MIT',
      packages=['terrainbento'],
      long_description=open('README.md').read(),
      zip_safe=False,
      install_requires=['scipy>=0.12',
                        'dill',
                        'numpy',
                        'landlab']
                        )
