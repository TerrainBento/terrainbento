#!/usr/bin/env bash
py.test terrainbento --doctest-modules --cov=terrainbento tests/ --cov-report term-missing
