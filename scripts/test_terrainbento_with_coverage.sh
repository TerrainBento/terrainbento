#!/usr/bin/env bash
pytest terrainbento --durations=0 --doctest-modules --cov=terrainbento tests/ --cov-report term-missing
