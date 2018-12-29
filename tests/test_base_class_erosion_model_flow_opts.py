# coding: utf8
# !/usr/env/python

import pytest

from landlab.components import (
    DepressionFinderAndRouter,
    FlowDirectorMFD,
    FlowDirectorSteepest,
)
from terrainbento import ErosionModel
from terrainbento.utilities import filecmp


def test_FlowAccumulator_with_depression_steepest(clock_simple):
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "flow_director": "FlowDirectorSteepest",
        "depression_finder": "DepressionFinderAndRouter",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)
    assert isinstance(
        em.flow_accumulator.depression_finder, DepressionFinderAndRouter
    )


def test_no_depression_finder(clock_simple):
    params = {"model_grid": "RasterModelGrid", "clock": clock_simple}

    em = ErosionModel(params=params)
    assert em.flow_accumulator.depression_finder is None


def test_FlowAccumulator_with_D8_Hex(clock_simple):
    params = {
        "model_grid": "HexModelGrid",
        "clock": clock_simple,
        "flow_director": "D8",
    }
    pytest.raises(NotImplementedError, ErosionModel, params=params)


def test_FlowAccumulator_with_depression_MFD(clock_simple):
    params = {
        "model_grid": "HexModelGrid",
        "clock": clock_simple,
        "flow_director": "MFD",
    }
    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorMFD)


def test_alt_names_steepest(clock_simple):
    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "flow_director": "D4",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)

    params = {
        "model_grid": "RasterModelGrid",
        "clock": clock_simple,
        "flow_director": "Steepest",
    }

    em = ErosionModel(params=params)
    assert isinstance(em.flow_accumulator.flow_director, FlowDirectorSteepest)
