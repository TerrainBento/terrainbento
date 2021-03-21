#! /usr/bin/env python

from setuptools import setup, find_packages
import pkg_resources

setup(
    name="terrainbento",
    version=pkg_resources.get_distribution("terrainbento").version,
    description="TerrainBento suite of landscape evolution models",
    url="https://github.com/TerrainBento/terrainbento/",
    author="The TerrainBento Team",
    author_email="barnhark@colorado.edu",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    packages=find_packages(),
    install_requires=open("requirements.txt", "r").read().splitlines(),
    package_data={"": ["tests/*txt", "data/*txt", "data/*asc", "data/*nc"]},
)
