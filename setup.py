#! /usr/bin/env python

from setuptools import setup, find_packages
import versioneer

setup(
    name="terrainbento",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="TerrainBento suite of landscape evolution models",
    url="https://github.com/TerrainBento/terrainbento/",
    author="The TerrainBento Team",
    author_email="barnhark@colorado.edu",
    license="MIT",
    long_description=open("README.md").read(),
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "jupyter",
        "holoviews",
        "pandas",
        "xarray",
        "dask",
        "landlab>=1.9",
    ],
    package_data={"": ["tests/*txt", "data/*txt", "data/*asc", "data/*nc"]},
)
