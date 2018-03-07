from setuptools import setup

import versioneer

setup(name='terrainbento',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='TerrainBento suite of landscape evolution models',
      url='https://github.com/TerrainBento/terrainbento/',
      author='The TerrainBento Team',
      author_email='barnhark@colorado.edu',
      license='MIT',
      packages=['terrainbento'],
      zip_safe=False)
